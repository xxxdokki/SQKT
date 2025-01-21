import sys
sys.path.append('/home/doyounkim/sqkt')

import pandas as pd
import numpy as np
import random, re, logging, itertools
import torch
from torch.utils.data import random_split, Dataset, DataLoader, Subset,  ConcatDataset
import torch.nn as nn
from transformers import BertTokenizer, BertModel, T5ForConditionalGeneration, RobertaModel, RobertaTokenizer
from sklearn.model_selection import train_test_split
from collections import defaultdict



device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device type: {device}")


class ProjectionLayer(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(ProjectionLayer, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.activation = nn.ReLU()


    def forward(self, x):
        x = x.view(-1, self.linear.in_features)
        return self.activation(self.linear(x))



class SkillDataset(Dataset):
    def __init__(self, code_file_path, text_file_path, help_center_path, target_dict, question_model, device, debug = False):
        """
        code_file_path: code submission data
        text_file_path: problem text data
        help_center_path: students' questions data
        target_dict: target data
        question_model: codet5 (shared with main.py file, main.py에서 codet5 auxiliary task loss계산을 하고 data_loader.py파일에서 임베딩을 해야하기 때문에)
        device: cuda or cpu
        debug: debug option
        """
        self.device = device
        
        #code submission data 
        self.code_df = pd.read_csv(code_file_path)

        #problem description data
        self.text_df = pd.read_csv(text_file_path)

        #students' questions data
        self.help_center_log = pd.read_csv(help_center_path)

        #target load
        self.target_dict = target_dict

        self.debug = debug


        torch.cuda.empty_cache()


        if self.debug:
            logging.info(f"\n\n Check data type \n\n.")
            logging.info(f"Data type of created_datetime: {self.code_df['created_datetime'].dtype}") #Data type of created_datetime: float64
            logging.info(f"First few values of created_datetime: {self.code_df['created_datetime'].head()}")


            # 숫자(타임스탬프)인지 확인
            if pd.api.types.is_numeric_dtype(self.code_df['created_datetime']):
                logging.info(f"created_datetime appears to be numeric (likely timestamp)")
                # 타임스탬프의 범위 확인 (밀리초 vs 초)
                min_timestamp = self.code_df['created_datetime'].min()
                max_timestamp = self.code_df['created_datetime'].max()
                logging.info(f"Timestamp range: {min_timestamp} to {max_timestamp}")
                
                if min_timestamp > 1e12:  
                    logging.info(f"Timestamps appear to be in milliseconds")
                else:
                    logging.info(f"Timestamps appear to be in seconds")
            else:
                logging.info(f"created_datetime appears to be non-numeric (likely string)")
                # 문자열 형식 샘플 출력
                logging.info("Sample datetime strings:")
                logging.info(f"{self.code_df['created_datetime'].sample(5).tolist()}")

            """
            Timestamp range: 1642043709105.688 to 1711014749722.469
            Timestamps appear to be in milliseconds
            """

            # 파싱 시도
            try:
                parsed_dates = pd.to_datetime(self.code_df['created_datetime']/1000, unit='s', utc=True)
                logging.info(f"Successfully parsed as datetime")
                logging.info(f"Parsed date range:")
                logging.info(f"Min: {parsed_dates.min()}")
                logging.info(f"Max: {parsed_dates.max()}")
            except ValueError as e:
                logging.info(f"Error parsing dates: {e}")

            """
            Successfully parsed as datetime
            Parsed date range:
            Min: 2022-01-13 03:15:09.105688064+00:00
            Max: 2024-03-21 09:52:29.722469120+00:00
            """

            logging.info(f"\n\n")

        
        """
        code submission data's time format: Unix time (밀리초 단위)
        desired output format: pandas Timestamp (UTC 시간대)
        1) 'created_datetime' 열의 밀리초 단위 Unix 타임스탬프를 pandas Timestamp 객체로 변환. 
        2) /1000을 통해 밀리초를 초 단위로 변환하고, unit='s'로 초 단위임을 지정. 
        3) utc=True로 UTC 시간대로 설정. 
        그 후 두 파일의 시간대 형식을 맞춤. -> 비교를 용이하게 하기 위하여.
        """
        self.code_df['created_datetime'] = pd.to_datetime(self.code_df['created_datetime'] /1000, unit='s', utc=True)
        self.help_center_log['post_created_datetime'] = pd.to_datetime(self.help_center_log['post_created_datetime'])
        self.code_df['created_datetime'] = self.code_df['created_datetime'].dt.tz_convert('UTC')
        self.help_center_log['post_created_datetime'] = self.help_center_log['post_created_datetime'].dt.tz_convert('UTC')
        # print(self.help_center_log['post_created_datetime'][:5])


        #tokenizer + model 
        self.code_tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
        self.text_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.question_tokenizer = RobertaTokenizer.from_pretrained("Salesforce/codet5p-220m")
        self.code_model = RobertaModel.from_pretrained("microsoft/codebert-base").to(device)
        self.text_model = BertModel.from_pretrained('bert-base-uncased').to(device)
        self.question_model = question_model.to(device) #question model의 경우 main.py에서 loss 계산해야하므로 두 파일에서 공유
        self.resize_token_embeddings() 


        #set projection layer
        self.projection_text = ProjectionLayer(768, 512).to(device) #shared between problem text and skill embedding
        self.projection_code = ProjectionLayer(768, 512).to(device)
        self.projection_qna = ProjectionLayer(768, 512).to(device)

        """
        'submission_order' 구조는 아래와 같음.
                                                                        x_user_id  exercise_id  submission_order
        0  40c258bc0488f4a4c10ba6ab7f249eff8992fd24317c45d81df1672685ced824      1548385                 1
        1  40c258bc0488f4a4c10ba6ab7f249eff8992fd24317c45d81df1672685ced824      1548385                 2
        2  40c258bc0488f4a4c10ba6ab7f249eff8992fd24317c45d81df1672685ced824      1548385                 3
        3  40c258bc0488f4a4c10ba6ab7f249eff8992fd24317c45d81df1672685ced824      1548385                 4
        4  40c258bc0488f4a4c10ba6ab7f249eff8992fd24317c45d81df1672685ced824      1548385                 5
        5  40c258bc0488f4a4c10ba6ab7f249eff8992fd24317c45d81df1672685ced824      1548385                 6
        6  40c258bc0488f4a4c10ba6ab7f249eff8992fd24317c45d81df1672685ced824      1548385                 7
        7  40c258bc0488f4a4c10ba6ab7f249eff8992fd24317c45d81df1672685ced824      1548415                 1
        8  40c258bc0488f4a4c10ba6ab7f249eff8992fd24317c45d81df1672685ced824      1548415                 2
        9  40c258bc0488f4a4c10ba6ab7f249eff8992fd24317c45d81df1672685ced824      1548415                 3
        -> 사용자는 1548385문제를 7번 제출했고, 그 다음 문제를 3+번 제출함.
        """
        self.code_df['submission_order'] = self.code_df.groupby(['x_user_id', 'exercise_id']).cumcount() + 1 # +1은 1번부터 시작하도록 하려고
        # if self.debug:
        #     print(self.code_df[['x_user_id', 'exercise_id', 'submission_order']].head(10).to_string(index=True))
        

        self.previous_user_id = None #이전에 처리한 사용자의 ID를 저장함. 사용자가 변경될 때마다 누적 임베딩을 초기화하는 데 활용.

        self.cumulative_embeddings = [] 
        #현재 사용자의 누적된 학습 과정의 임베딩을 저장. 사용자의 이전 문제 해결 기록과 현재 문제의 정보를 누적하여 저장하는 용도.
        #이를 통해 사용자의 학습 진행 상황을 시간 순서대로 추적할 수 있음. 
        #따라서 앞 배치와 사용자가 동일하다면 self.cumulative 는 앞 배치것에 이어서 작성됨.

        self.previous_problem_id = None #이전에 처리한 문제의 ID를 저장. 문제가 변경될 때마다 필요한 처리(예: 새로운 문제 설명 임베딩 추가)를 수행하는 데 활용.
        self.first_call = True
        """
        self.user_problem_map의 구성 
        {
            '사용자ID1': {
                문제ID1: [제출인덱스1, 제출인덱스2, ...],
                문제ID2: [제출인덱스1, 제출인덱스2, ...],
                ...
            },
            '사용자ID2': {
                문제ID1: [제출인덱스1, 제출인덱스2, ...],
                문제ID2: [제출인덱스1, 제출인덱스2, ...],
                ...
            },
            ...
        }

        """
        self.user_problem_map = self._create_user_problem_map()

        if self.debug:
            first_10_items = dict(itertools.islice(self.user_problem_map.items(), 10))
            logging.info(f"User problem map samples: {first_10_items}\n\n")
            """
            {
            '40c258bc0488f4a4c10ba6ab7f249eff8992fd24317c45d81df1672685ced824': defaultdict(<class 'list'>, 
                {1548385: [0, 1, 2, 3, 4, 5, 6], 
                1548415: [7, 8, 9], 
                1548498: [10, 11, 12, 13, 14, 15, 16, 17, 18, 19], 
                1549215: [20, 21]}), 
            '5d39c1f4640c3cdd6c4bc5ac302a791299a6f2812312bd0eeaef64a246e034f4': defaultdict(<class 'list'>, 
                {1548498: [22, 23, 24, 25, 26], 
                1548709: [27, 28, 29, 30, 31, 32, 33, 34], 
                1557658: [35, 36], 
                1557778: [37, 38, 39], 
                1557907: [40, 41, 42, 43, 44, 45, 46, 47, 48]}), 
            '3addc2ea41ec032df9eca1b663d2b62d3b0fc65f7db0f72c21a88cea2ed4fdb7': defaultdict(<class 'list'>, 
                {1548498: [49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69], 
                1548709: [70]}), 
            'c57caa469e36a8b8ca951fe44883cc84827e59b1618492aa934bedfa6ca531a9': defaultdict(<class 'list'>, 
                {1548415: [71, 72], 
                1551598: [73, 74], 
                1551714: [75, 76], 
                1554876: [77, 78]}), 
            '95481986af69e048d9b46d33e942076809a770c867f0c7243f638bc23a69ba43': defaultdict(<class 'list'>, 
                {1548385: [79, 80, 81], 
                1548498: [82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134], 1558476: [135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182], 
                1559312: [183, 184]}), 
            '72d00c455d2b166025dd8b63c1006a3d16c324d871221e22ba7379ad55cd01b9': defaultdict(<class 'list'>, 
                {1548498: [185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205], 
                1548709: [206, 207, 208, 209, 210, 211, 212], 
                1549445: [213, 214, 215, 216, 217, 218], 
                1550975: [219, 220], 
                1551307: [221, 222, 223], 
                1551465: [224, 225], 
                1558476: [226], 
                1558550: [227], 
                1559424: [228, 229]}), 
            }
            """

        self.previous_codet5_input_ids = None
        self.previous_codet5_attention_mask = None
        self.previous_codet5_targets = None



    def resize_token_embeddings(self):
        """
        Special tokens [SKILL], [TARGET] 등을 토크나이저에 추가.
        """
        special_tokens_dict = {'additional_special_tokens': ['[SKILL]', '[TARGET SKILL]', '[QUESTION]', '[TARGET]', '[CODE]', '[TEXT]']}
        self.text_tokenizer.add_special_tokens(special_tokens_dict)
        self.code_tokenizer.add_special_tokens(special_tokens_dict)
        self.question_tokenizer.add_special_tokens(special_tokens_dict)

        self.text_model.resize_token_embeddings(len(self.text_tokenizer))
        self.code_model.resize_token_embeddings(len(self.code_tokenizer))
        self.question_model.resize_token_embeddings(len(self.question_tokenizer))


    def __len__(self):
        #각 학생이 푼 모든 문제에 대해 마지막 제출을 제외한 제출 내역의 수를 합산
        #마지막 제출은 학습에 사용하지 않고 테스트 또는 평가에 사용하려는 목적
        length = sum(len(problems) - 1 for problems in self.user_problem_map.values())
        return length

        
    def __getitem__(self, idx):
            """
            Args:
                idx (int): 데이터셋에서 가져올 항목의 인덱스

            Returns:
                tuple: 다음 요소를 포함하는 튜플:
                    - cumulative_embeddings (torch.Tensor): 누적된 임베딩 텐서 (학생 교육 history). Shape:[N, 512] (N은 누적된 길이)
                    - target (int): 주어진 사용자와 문제에 대한 타겟 값. Shape: scalar
                    - anchor_embeddings (torch.Tensor): 현재 문제에 대한 앵커 임베딩. Shape: [M, 512] (M은 현재 문제에 대한 제출 횟수)
                    - positive_emb (torch.Tensor): 긍정 샘플에 대한 임베딩. Shape: [M, 512]
                    - negative_emb (torch.Tensor): 부정 샘플에 대한 임베딩. Shape: [1, 512]
                    - positive_qna_emb (torch.Tensor): 긍정 질문-응답 샘플에 대한 임베딩. Shape: [1, 512]
                    - negative_qna_emb (torch.Tensor): 부정 질문-응답 샘플에 대한 임베딩. Shape: [1, 512]
                    - codet5_input_ids (torch.Tensor): Codet5 모델에 대한 입력 ID. Shape: [P, 512] (P는 질문-응답 쌍의 수)
                    - codet5_attention_mask (torch.Tensor): Codet5 모델에 대한 어텐션 마스크. Shape: [P, 512]
                    - codet5_targets (torch.Tensor): Codet5 모델에 대한 타겟 ID. Shape: [P, 512]
                    - original_data (dict): 원본 데이터에 대한 정보가 담긴 딕셔너리 
            """
            #one batch : 하나의 학생이 하나의 문제를 푼 정보로 구성
            user_idx, problem_idx = self._get_indices(idx)
            
            """
            'da13b1835a64ace869a29a0ebc54f43befe8a64961d37798cb4f99ec1b183a6b': defaultdict(<class 'list'>, 
            {
                1551714: [7567, 7568, 7569, 7570, 7571, 7572], 
                1554364: [7573, 7574]
            })
            (user_idx, problem_idx)가 (0, 0) 이라면 
                user_id = 'da13b1835a64ace869a29a0ebc54f43befe8a64961d37798cb4f99ec1b183a6b'
                problem_list = [1551714, 1554364]
                problem_id = 1551714 

            (user_idx, problem_idx)가 (0, 1) 인 경우,
                user_id = 'da13b1835a64ace869a29a0ebc54f43befe8a64961d37798cb4f99ec1b183a6b'
                problem_list = [1551714, 1554364]
                problem_id = 1554364
            """
            user_id = list(self.user_problem_map.keys())[user_idx]
            problem_list = list(self.user_problem_map[user_id].keys())
            problem_id = problem_list[problem_idx]


            #새로운 사용자를 감지하고 누적 임베딩을 초기화하는 역할
            if self.previous_user_id != user_id:
                self.cumulative_embeddings = []
                if self.debug:
                    logging.info("\n\nNew user detected. Resetting cumulative embeddings.")

            # 사용자가 바뀌거나 같은 사용자의 다른 문제로 넘어갈 때 실행
            # 새로운 문제에 대한 텍스트 임베딩을 생성 후 cumulative_embeddings에 append
            if self.previous_user_id != user_id or self.previous_problem_id != problem_id:
                text_embedding = self._embed_text(problem_id, next = False)
                self.cumulative_embeddings.append(text_embedding)
                if self.debug:
                    logging.info(f"\n\nNew problem detected. Adding text embedding to cumulative embeddings.")

                

            #학생의 현재 문제에 대한 제출 index 구함
            submission_indices = self.user_problem_map[user_id][problem_id] #ex. [20, 21]

            if self.debug:
                logging.info(f"\n\nuser_id: {user_id}, problem_id: {problem_id}")
                logging.info(f"submission_indices: {submission_indices}")
                
            code_embeddings = [] #code submissions 저장
            question_embeddings = [] #question이 있다면 저장 없으면 (1, 512)
            current_problem_embeddings = [] #나중에 anchor가 됨
            codet5_input_ids_list = [] #codet5 fine-tuning을 위한 학생 질문 토큰 저장
            codet5_attention_mask_list = []
            codet5_targets_list = [] #codet5 fine-tuning을 위한 교사 응답 토큰 저장

            flag = True #학생 질문이 없을 경우 사용하는 flag

            #현재 사용자가 한 문제에 대해 푼 모든 submission에 대해 반복
            for i, submission_idx in enumerate(submission_indices):
                if isinstance(submission_idx, int):
                    row = self.code_df.iloc[submission_idx]
                else:
                    row = self.code_df.iloc[submission_idx[1]]  #지우는거 고민

                code_embedding = self._embed_code(row['contents']) #[1, 512]
                code_embeddings.append(code_embedding) 
                #이 리스트는 한 학생이 해당 문제에 대해 여러 번 제출한 모든 코드 임베딩을 담게 됨. 따라서 for 문을 다 돌면 제출 횟수만큼 길어짐. 
                current_problem_embeddings.append(code_embedding) 

                current_submission_time = row['created_datetime']
                
                if i > 0:  # 첫 번째 제출이 아닌 경우
                    prev_submission_idx = submission_indices[i-1]
                    if isinstance(prev_submission_idx, int):
                        prev_submission_time = self.code_df.iloc[prev_submission_idx]['created_datetime']
                    else:
                        prev_submission_time = self.code_df.iloc[prev_submission_idx[1]]['created_datetime']
                else:  # 첫 번째 제출인 경우
                    prev_submission_time = pd.Timestamp.min
                

                # 시간 형식 맞추기
                if current_submission_time.tzinfo is None:
                    current_submission_time = current_submission_time.tz_localize('UTC')
                if prev_submission_time.tzinfo is None:
                    prev_submission_time = prev_submission_time.tz_localize('UTC')

                #현재 제출을 하기 전에 했던 질문 내역을 확인
                help_center_questions = self.help_center_log[
                    (self.help_center_log['x_user_id'] == user_id) &
                    (self.help_center_log['exercise_id'] == problem_id) &
                    (self.help_center_log['post_created_datetime'] > prev_submission_time) &
                    (self.help_center_log['post_created_datetime'] <= current_submission_time)
                ]
                
                
                if not help_center_questions.empty:
                    #질문이 있는 경우
                    if self.debug:
                        logging.info(f"\n\n============================================================")
                        logging.info(f"There is questions.")
                    questions, codet5_input_ids, codet5_attention_mask, codet5_targets = self._embed_questions_with_codet5(help_center_questions)
                    # questions = [#of questions, 512] or [1, 512]
                    question_embeddings.append(questions)
                    skills = self._embed_skills(help_center_questions)



                    question_embeddings.append(skills)
                    codet5_input_ids_list.append(codet5_input_ids)
                    codet5_attention_mask_list.append(codet5_attention_mask)
                    codet5_targets_list.append(codet5_targets)
                    self.previous_codet5_input_ids = codet5_input_ids #캐시 저장
                    self.previous_codet5_attention_mask = codet5_attention_mask
                    self.previous_codet5_targets = codet5_targets
                else:
                    flag = False
                    #질문을 하지 않은 경우
                    question_embeddings.append(torch.zeros(1, 512).to(self.device))
                    
                    # Random fallback 질문을 가져오는 로직 추가
                    # 질문이 없는 상황에서도 codet5는 계속 학습되어야 하기 때문에 랜덤한 질문과 그 질문에 대한 답변을 가져옴
                    # 학생의 submission 횟수만큼 반복하기 때문에 랜덤한 경우라면, 즉 submission과 무관한 경우라면 그것들 중에서 하나만 모델에게 보냄.
                    random_question, random_answer = self._fetch_random_question()
                    skills = self._embed_skills(random_question)
                    if self.debug:
                        logging.info(f"\n\n============================================================")
                        logging.info(f"There is NO question. So we made random questions.")
                        logging.info(f"random_question: {random_question}")
                        logging.info(f"random_answer: {random_answer}")
                        
            
            code_embeddings = torch.cat(code_embeddings, dim=0) #[# of code submission, 512]
            question_embeddings = torch.cat(question_embeddings, dim=0) #[# of questions , 512] or [# of code submission, 512] if no question asked
            # skill 개수도 고려해서 디멘젼
            anchor_embeddings = torch.cat(current_problem_embeddings, dim=0) #[#of code submission, 512]

            #codet5는 배치 안에 있는 각각의 질문을 독립적으로 인코딩->따라서 위아래로 쌓는 구조 가능
            pad_tensor = torch.full((1, 512), self.question_tokenizer.pad_token_id, dtype=torch.long).to(self.device)
            if flag == True:
                # 질문이 있는 경우, codet5를 학습하는 질문-응답 쌍도 해당 code submission과 관련이 있으므로 전부 다 사용
                codet5_input_ids = torch.cat(codet5_input_ids_list, dim=0) if codet5_input_ids_list else pad_tensor #[#of questions, 512] or [1, 512]
                codet5_attention_mask = torch.cat(codet5_attention_mask_list, dim=0) if codet5_attention_mask_list else pad_tensor
                codet5_targets = torch.cat(codet5_targets_list, dim=0) if codet5_targets_list else pad_tensor
            else:
                # 질문이 없는 경우, codet5를 학습하는 질문-응답 쌍은 해당 code submission과 관련이 없으므로 맨 앞의 것만 활용 
                codet5_input_ids = codet5_input_ids_list[0] if codet5_input_ids_list else pad_tensor #[#of questions, 512] or [1, 512]
                codet5_attention_mask = codet5_attention_mask_list[0] if codet5_attention_mask_list else pad_tensor
                codet5_targets = codet5_targets_list[0] if codet5_targets_list else pad_tensor
                # print("random shape: ", codet5_input_ids.shape)
                # 따라서 항상 [1, 512] shape

            
            self.cumulative_embeddings.append(code_embeddings)
            self.cumulative_embeddings.append(question_embeddings)

            next_problem_id = self._get_next_problem_id(user_id, problem_idx)
            next_problem_emb = self._embed_text(next_problem_id, next = True) if next_problem_id is not None and self.text_df['exercise_id'].eq(next_problem_id).any() else None
            next_problem_skill = self._embed_next_skills(next_problem_id) if next_problem_id is not None and self.text_df['exercise_id'].eq(next_problem_id).any() else None

            self.cumulative_embeddings.append(next_problem_emb) if next_problem_emb is not None else None
            self.cumulative_embeddings.append(next_problem_skill) if next_problem_skill is not None else None
            cumulative_embeddings = torch.cat(self.cumulative_embeddings, dim=0)
            # print("cumulative_embeddings: ", cumulative_embeddings.shape)

            # cumulative embeddings에 들어가야 하는 모든 데이터가 다 들어갔으므로, 업데이트 
            self.previous_user_id = user_id
            self.previous_problem_id = problem_id
            
            if self.debug == True:
                logging.info(f"Current user_id: {user_id}, problem_id: {problem_id}, next_problem_id: {next_problem_id}")
                print(f"Cumulative Embeddings type: {cumulative_embeddings.dtype}")
                logging.info(f"Anchor Embeddings Shape: {anchor_embeddings.shape}")
                print(f"Anchor Embeddings type: {cumulative_embeddings.dtype}")
                logging.info(f"Anchor Embeddings type: {cumulative_embeddings.dtype}")
                logging.info(f"Submissions for current problem: {len(submission_indices)}")
                logging.info(f"how many question embeddings: {len(question_embeddings)}")
                logging.info(f"then how many real code submissions: {len(anchor_embeddings)}")
                logging.info(f"Cumulative Embeddings Shape: {cumulative_embeddings.shape}")
                logging.info(f"codet5_input_ids: {codet5_input_ids.shape}")
                logging.info(f"codet5_input_ids samples: {codet5_input_ids[0]}")
                logging.info(f"codet5_attention_mask: {codet5_attention_mask.shape}")
                logging.info(f"codet5_targets: {codet5_targets.shape}")
                logging.info(f"Final codet5_input_ids dtype: {codet5_input_ids.dtype}")
                codet5_input_ids = codet5_input_ids.long()
                output = self.question_model.generate(
                    input_ids=codet5_input_ids,
                    attention_mask=codet5_attention_mask,
                    max_length=100,
                    num_return_sequences=1,
                    no_repeat_ngram_size=2,
                    do_sample=True,
                    top_k=50,
                    top_p=0.95,
                    temperature=0.7
                )
                decoded = self.question_tokenizer.decode(output[0], skip_special_tokens=True)
                logging.info(f"Tokenized output: {output[0]}")
                logging.info(f"Decoded output: {decoded}")
                logging.info(f"what is the decoded question?: {decoded}")
                codet5_input_ids = codet5_input_ids.long()
                codet5_targets = codet5_targets.long()

                # codet5 loss 계산
                codet5_loss = self.question_model(input_ids=codet5_input_ids, attention_mask=codet5_attention_mask, labels=codet5_targets).loss
                print(f"codet5_loss shape: {codet5_loss.shape}")
                logging.info(f"codet5_loss shape: {codet5_loss.shape}")
                print(f"codet5_loss type: {codet5_loss.dtype}")
                logging.info(f"codet5_loss type: {codet5_loss.dtype}")


            positive_emb = self._embed_text(problem_id).cpu()
            negative_problem_id = self._get_negative_problem_id(user_id, problem_id)
            negative_emb = self._embed_text(negative_problem_id).cpu()
            positive_qna_emb = self._get_question_positive_sample(user_id, problem_id).cpu()
            negative_qna_emb = self._get_question_negative_sample(user_id, problem_id).cpu()

            original_data = {
                'user_id': user_id,
                'problem_id': problem_id,
                'problem_order': problem_idx,
                'code_submissions': self._get_code_submissions(user_id, problem_id),
                'problem_text': self._get_problem_text(problem_id),
                'help_center_questions': self._get_question_text(user_id, problem_id),
                'skills' : skills
            }

            return cumulative_embeddings, self.target_dict.get((user_id, problem_id), 0), anchor_embeddings, positive_emb, negative_emb, positive_qna_emb, negative_qna_emb, codet5_input_ids, codet5_attention_mask, codet5_targets, original_data

        
    def _get_indices(self, idx):
        """
        self.user_problem_map = {
            'user1': {1001: [...], 1002: [...], 1003: [...]},
            'user2': {2001: [...], 2002: [...]},
            'user3': {3001: [...], 3002: [...], 3003: [...], 3004: [...]}
        }
        1. _get_indices(0):

        count = 0, user_idx = 0, problem_idx = 0 (user1의 1001)
        count == idx (0 == 0), 따라서 (0, 0) 반환


        2. _get_indices(2):

        count = 0, user_idx = 0, problem_idx = 0 (user1의 1001)
        count = 1, user_idx = 0, problem_idx = 1 (user1의 1002)
        count = 2, user_idx = 0, problem_idx = 2 (user1의 1003)
        count == idx (2 == 2), 따라서 (0, 2) 반환


        3. _get_indices(3):

        user1의 모든 문제 처리 (count = 0, 1, 2)
        count = 3, user_idx = 1, problem_idx = 0 (user2의 2001)
        count == idx (3 == 3), 따라서 (1, 0) 반환


        4. _get_indices(5):

        user1의 모든 문제 처리 (count = 0, 1, 2)
        user2의 모든 문제 처리 (count = 3, 4)
        count = 5, user_idx = 2, problem_idx = 0 (user3의 3001)
        count == idx (5 == 5), 따라서 (2, 0) 반환


        5. _get_indices(8):

        user1의 모든 문제 처리 (count = 0, 1, 2)
        user2의 모든 문제 처리 (count = 3, 4)
        user3의 모든 문제 처리 (count = 5, 6, 7, 8)
        count == idx (8 == 8), 따라서 (2, 3) 반환


        6. _get_indices(9):

        모든 사용자와 문제를 순회
        마지막 count는 8
        IndexError("Index 9 out of range") 발생
        """
        
        count = 0
        users_list = list(self.user_problem_map.keys()) 
        
        for i, (user_id, problems) in enumerate(self.user_problem_map.items()):
            start_count = count  # 현재 유저의 시작 인덱스

            for problem_idx, problem_id in enumerate(problems.keys()):
                if count == idx:
                    if hasattr(self, 'first_call') and self.first_call:  # 첫 호출일 때만 체크
                        self.first_call = False  # 플래그 해제
                        if start_count != idx:  # 중간 문제라면
                            # 다음 유저의 첫 번째 문제 찾기
                            for next_i in range(i + 1, len(users_list)):
                                next_user = users_list[next_i]
                                if self.user_problem_map[next_user]:
                                    return next_i, 0
                    return i, problem_idx 
                count += 1

        raise IndexError(f"Index {idx} out of range")


    def _get_next_problem_id(self, user_id, current_problem_idx):
        #특정 학생(user_id)이 현재 푸는 문제(current_problem_idx) 다음 문제의 ID를 반환하는 함수
        problem_list = list(self.user_problem_map[user_id].keys())
        if current_problem_idx < len(problem_list) - 1:
            return problem_list[current_problem_idx + 1]
        return None

    def _embed_code(self, content):
        """
        학생의 코드 제출 contents를 임베딩하는 함수

        Args:
            content (str): 임베딩할 코드 내용이 포함된 문자열

        Returns:
            torch.Tensor: 주어진 코드 내용에 대한 임베딩 벡터. Shape: [1, 512]
        """
        token_types = '[CODE]'
        content = token_types + ' ' + str(content)
        code_tokens = self.code_tokenizer.tokenize(content)
        tokens = [self.code_tokenizer.cls_token] + code_tokens + [self.code_tokenizer.eos_token]
        tokens_ids = self.code_tokenizer.convert_tokens_to_ids(tokens)

        max_length = 256
        if len(tokens_ids) > max_length:
            tokens_ids = tokens_ids[:max_length]
        else: #0이 무슨 토큰인지 
            tokens_ids = tokens_ids + [0] * (max_length - len(tokens_ids))

        with torch.no_grad(): # None 체크
            embedding = self.code_model(torch.tensor(tokens_ids)[None, :].to(self.device)).last_hidden_state.max(dim=1).values

        return self.projection_code(embedding)
    
    

    def _embed_text(self, ex_id, next = True):
        """
        주어진 문제 ID에 대한 description을 임베딩하는 함수

        Args:
            ex_id (int): 임베딩할 문제의 ID.
            next (bool): 다음 문제 정보를 사용할지 여부. True이면 '[TARGET]' 토큰 타입을 사용하고, False이면 '[TEXT]' 토큰 타입을 사용.

        Returns:
            torch.Tensor: 주어진 문제 내용에 대한 임베딩 벡터. Shape: [1, 512]
        """
        # next == True이면 다음 문제 정보 (토큰 타입 분리를 위함)
        if next:
            max_length = 512
            token_type = '[TARGET]'
            text = self.text_df.loc[self.text_df['exercise_id'] == ex_id, ['Instruction Content', 'Solution Content']].values[0]
            text = ' '.join(str(t) for t in text)
            text = token_type + ' ' + text

            tokens_ids = self.text_tokenizer.encode(text, add_special_tokens=False, max_length=max_length-2, truncation=True)
            input_ids = torch.tensor([self.text_tokenizer.cls_token_id] + tokens_ids + [self.text_tokenizer.sep_token_id]).unsqueeze(0).to(self.device)

            with torch.no_grad():
                outputs = self.text_model(input_ids)
                embeddings = outputs.last_hidden_state.max(dim=1).values

            return self.projection_text(embeddings)
        else:
            max_length = 512
            token_type = '[TEXT]'
            text = self.text_df.loc[self.text_df['exercise_id'] == ex_id, ['Instruction Content', 'Solution Content']].values[0]
            text = ' '.join(str(t) for t in text)
            text = token_type + ' ' + text
            tokens_ids = self.text_tokenizer.encode(text, add_special_tokens=False, max_length=max_length-2, truncation=True)
            input_ids = torch.tensor([self.text_tokenizer.cls_token_id] + tokens_ids + [self.text_tokenizer.sep_token_id]).unsqueeze(0).to(self.device)

            with torch.no_grad():
                outputs = self.text_model(input_ids)
                embeddings = outputs.last_hidden_state.max(dim=1).values

            return self.projection_text(embeddings)
        
    def _embed_questions_with_codet5(self, help_center_questions):
        """
        Args: help_center_questions (해당 제출물을 제출하기 전에 했던 질문 내역들)

        Returns: 
            if 질문-대답 쌍이 일정할 경우:
                질문 임베딩([질문 개수, 512]), codet5_input_ids([질문 개수, 512]), codet5_attention_masks([질문 개수, 512]) , codet5_targets([질문 개수, 512]) 
            else:
                질문 임베딩([질문 개수, 512]), codet5_input_ids([1, 512]), codet5_attention_masks([1, 512]) , codet5_targets(1, 512]) 
        """
        question_embeddings = [] #질문 임베딩 담는 리스트
        codet5_input_ids_list = [] #codet5 auxiliary task를 위한 질문 토큰
        codet5_attention_mask_list = [] #codet5 auxiliary task를 위한 질문 attention mask
        codet5_targets_list = [] #codet5 auxiliary task를 위한 응답 토큰

        for _, question in help_center_questions.iterrows():
            # 학생의 질문인 경우
            if question['is_student'] == True:
                question_text = f"question: {question['content']}"
                if self.debug:
                    logging.info(f"\n\nwhat is the question?: {question_text}")
                tokenized = self.question_tokenizer(question_text, return_tensors="pt", padding="max_length", truncation=True, max_length=512)
                inputs = tokenized.input_ids.to(self.device)
                attention_mask = tokenized.attention_mask.to(self.device)
                
                with torch.no_grad():
                    embedding = self.question_model.encoder(input_ids=inputs, attention_mask=attention_mask).last_hidden_state.mean(dim=1)
                    # embedding: [1, 512]
                    question_embeddings.append(embedding) #질문 임베딩
                
                codet5_input_ids_list.append(inputs) #토큰화된 질문 for fine-tuning
                codet5_attention_mask_list.append(attention_mask) #attention mask
            # 튜터의 답변인 경우
            else:
                target_text = question['content']  # assuming the ground truth is stored in 'content'
                targets = self.question_tokenizer(target_text, return_tensors="pt", padding="max_length", truncation=True, max_length=512).input_ids.to(self.device)
                codet5_targets_list.append(targets)
                

        # #학생의 질문이 없거나 교사의 응답이 없을 경우 -> codet5 fine tuning과 관련된 모든걸 초기화하고 랜덤 질문 추가
        if not codet5_input_ids_list or not codet5_targets_list:
            codet5_input_ids_list = []
            codet5_attention_mask_list = []
            codet5_targets_list = []

            # Random fallback 질문을 가져오는 함수
            random_question, random_answer = self._fetch_random_question()

            random_question_text = f"question: {random_question}"
            tokenized = self.question_tokenizer(random_question_text, return_tensors="pt", padding="max_length", truncation=True, max_length=512)
            inputs = tokenized.input_ids.to(self.device)
            attention_mask = tokenized.attention_mask.to(self.device)

            codet5_input_ids_list.append(inputs)
            codet5_attention_mask_list.append(attention_mask)
                
            random_answer_text = f"answer: {random_answer}"
            tokenized_ans = self.question_tokenizer(random_answer_text, return_tensors='pt', padding="max_length", truncation=True, max_length=512)
            target_inputs = tokenized_ans.input_ids.to(self.device)
            codet5_targets_list.append(target_inputs)
            if self.debug:
                logging.info(f"No match. Random fetch.")
                logging.info(f"question: {random_question_text}")
                logging.info(f"answer: {random_answer_text}")

        # 길이 맞추기
        max_length = max(len(codet5_input_ids_list), len(codet5_targets_list))
        for list_to_pad in [codet5_input_ids_list, codet5_attention_mask_list, codet5_targets_list]:
            while len(list_to_pad) < max_length:
                list_to_pad.append(torch.full((1, 512), self.question_tokenizer.pad_token_id, dtype=torch.long).to(self.device))


        pad_tensor = torch.full((1, 512), self.question_tokenizer.pad_token_id, dtype=torch.long).to(self.device)
        # 결과 처리
        if question_embeddings:
            question_embeddings = self.projection_qna(torch.cat(question_embeddings, dim=0)) #[# of questions, 512]
        else:
            question_embeddings = pad_tensor

        if codet5_input_ids_list:
            codet5_input_ids = torch.cat(codet5_input_ids_list, dim = 0) #[# of questions, 512]
        else:
            codet5_input_ids = pad_tensor

        if codet5_attention_mask_list:
            codet5_attention_mask = torch.cat(codet5_attention_mask_list, dim = 0) #[# of questions, 512]
        else:
            codet5_attention_mask = pad_tensor

        if codet5_targets_list:
            codet5_targets = torch.cat(codet5_targets_list, dim = 0) #[# of answers, 512]
        else:
            codet5_targets = pad_tensor

        return question_embeddings, codet5_input_ids, codet5_attention_mask, codet5_targets

    def _fetch_random_question(self):
        """
        Fetch a random student question and its corresponding teacher answer from the dataset.

        Returns: 랜덤으로 선택한 학생의 질문과 그에 대한 교사의 응답 (string)
        """
        #학생의 질문만 남도록 필터링
        student_questions = self.help_center_log[self.help_center_log['is_student'] == True]

        
        if not student_questions.empty:
            random_index = random.randint(0, len(student_questions) - 1) #질문 랜덤 샘플링
            random_question_row = student_questions.iloc[random_index]
            random_question = random_question_row['content'] 
            exercise_id = random_question_row['exercise_id'] #해당 질문에 대한 응답 찾기 위해
            post_time = random_question_row['post_created_datetime'] #해당 질문에 대한 응답 찾기 위해


            # 관련된 교사 응답 패칭 (학생의 질문 다음에 오는 교사의 응답 중 첫번째 것을 가져오려고 함)
            teacher_answers = self.help_center_log[
                (self.help_center_log['exercise_id'] == exercise_id) &
                (self.help_center_log['is_student'] == False) &
                (self.help_center_log['post_created_datetime'] >= post_time) 
            ]

            if not teacher_answers.empty:
                teacher_answer = teacher_answers.iloc[0]['content']
            else:
                teacher_answer = "No corresponding teacher response found."  #없는 경우 


            return random_question, teacher_answer

        #학생의 질문이 없는 경우는 없지만 혹시 모르므로..
        return None, None

    def _embed_question(self, question_text):
        """
        주어진 질문 텍스트를 codet5 모델을 통해 임베딩하는 함수.

        Args:
            question_text (str): 질문 내용이 포함된 문자열.

        Returns:
            torch.Tensor: codet5 모델을 통해 생성된 질문 임베딩. 

        """
        max_length = 512
        question_text = f"question: {question_text}"
        
        # 토큰화 및 입력 데이터 생성
        inputs = self.question_tokenizer(question_text, return_tensors="pt", padding="max_length", truncation=True, max_length=max_length).input_ids.to(self.device)
        attention_mask = self.question_tokenizer(question_text, return_tensors="pt", padding="max_length", truncation=True, max_length=max_length).attention_mask.to(self.device)
        
        with torch.no_grad():
            # codet5 모델을 사용하여 임베딩 생성
            embedding = self.question_model.encoder(input_ids=inputs, attention_mask=attention_mask).last_hidden_state.mean(dim=1)
            # if self.debug:
            #     logging.info(f"what is the codet5 embedding shape?: {embedding.shape}")
        
        return self.projection_qna(embedding)
    
    def _embed_skills(self, questions):
        """
        주어진 질문 데이터에서 스킬을 추출하고 (_process_skills 함수 활용) 임베딩 만듦.

        Args:
            questions (str, pd.DataFrame, list): 질문 데이터는 다양한 형식으로 제공됨 (질문 데이터가 있는 경우 DataFrame, 없는 경우 string)

        Returns:
            추출된 스킬의 임베딩이 모두 포함된 텐서. 위아래로 연결됨.
            스킬이 없을 경우 [1, 768] 크기의 0으로 채워진 텐서를 반환. 
            나중에 projection layer를 통과하면서 [1, 512] 크기가 됨
        
        동작 과정:
            1. 입력된 질문이 문자열인 경우, 리스트로 변환하여 일관되게 처리.
            2. 질문이 판다스 DataFrame인 경우, 각 행의 'content'에서 스킬을 추출하고 임베딩 생성.
            3. 질문이 리스트인 경우, 리스트의 각 항목에서 스킬을 추출하고 임베딩 생성.
            4. 모든 스킬 임베딩을 텐서로 결합하여 반환. 만약 스킬 임베딩이 없으면 [1, 768] 크기의 0 텐서를 반환.
        """
        skill_embeddings = []

        if isinstance(questions, str):
            questions = [questions] 

        if isinstance(questions, pd.DataFrame):
            for _, question in questions.iterrows(): 
                content = str(question['content'])
                embeddings = self._process_skill(content) #skill extract
                skill_embeddings.append(embeddings)
        

        elif isinstance(questions, list):
            for question in questions: 
                embeddings = self._process_skill(question)
                skill_embeddings.append(embeddings)

        skill_embeddings = torch.cat(skill_embeddings, dim=0) if skill_embeddings else torch.zeros(1, 768).to(self.device)

        return skill_embeddings


    def _process_skill(self, content):
        """
        주어진 질문(str)에서 스킬을 추출하고 임베딩을 생성하는 보조 함수.

        Args:
            content(str): skill을 추출해야 하는 문장

        Returns:
            추출된 스킬을 임베딩한 텐서. 스킬이 없을 경우 기본 임베딩을 반환. Shape: [1, 512]

        동작 과정:
            1. 주어진 콘텐츠에서 코드 블락과 에러 메시지를 정규 표현식을 사용하여 추출.
            2. 코드와 에러 메시지를 제거한 후 남은 텍스트를 기반으로 분석 유형을 결정.
            3. 추출된 코드, 에러, 텍스트, 분석 유형을 기반으로 스킬을 추출 (_extract_skills 함수 사용)
            4. 추출된 스킬을 공백으로 구분된 하나의 문자열로 결합한 후, 이를 임베딩 벡터로 변환.
        """
        
        text = content.strip()
        # 에러 패턴과 코드 패턴
        error_pattern = r'(\w+Error): (.+)'  # error pattern
        code_pattern = r'```(.*?)```'  # code block pattern

        # 에러 메시지를 먼저 추출
        errors = re.findall(error_pattern, content, re.DOTALL)

        # 에러 메시지를 제거한 나머지 텍스트에서 코드 블록을 추출
        remaining_content = re.sub(error_pattern, '', content, re.DOTALL)
        codes = re.findall(code_pattern, remaining_content, re.DOTALL)

        
        if "설명해" in content or "문제" in content or "설명해" in content:
            analysis = "Explanation needed"
        elif "버그" in content:
            analysis = "Bug fixing"
        elif "번역" in content:
            analysis = "Translation"
        elif "리팩터" in content:
            analysis = "Refactoring"
        elif "시간복잡도" in content:
            analysis = "Time complexity"
        elif "의도" in content:
            analysis = "Intent understanding"
        else:
            analysis = "Unknown"

        #_extract_skills로 들어가기 위한 형식 맞추기
        codes = self._parse_content_column(codes)
        errors = self._parse_content_column(errors)
        text = self._parse_content_column(text)
        analysis = self._parse_content_column(analysis)

        #self._extract_skills()를 호출하여 스킬 추출
        #skills (list)
        skills = self._extract_skills(codes, errors, text, analysis)

        if self.debug:
            logging.info(f"Extracted skills: {skills}")

        
        skill_text = ' '.join(skills)

        #임베딩
        embedding = self._embed_skill(skill_text)

        return embedding #string 하나만 파라미터로 전달받았기 때문에 임베딩 하나만 리턴
    

    def _parse_content_column(self, content):
        """
        스킬 추출 등 후속 작업에 필요한 리스트 형식의 데이터를 준비하기 위한 함수
        """
        if isinstance(content, str):
            content = content.strip("[]").replace("'", "").split(", ")
        return content if isinstance(content, list) else [str(content)]
    

    


    def _add_skill(self, skills, skill):
        if skill not in skills:
            skills.append(skill)

    def _extract_skills(self, codes, errors, texts, analysis):
        """
        모델의 skill extractor system을 담당하는 함수

        Args:
            codes (list): 코드 블록 목록 
            errors (list): 에러 메시지 목록 
            texts (list): 텍스트 목록
            analysis (list): analysis type 목록

        Returns:
            list: 추출된 스킬의 목록을 반환. 중복되는 스킬은 제거됨.
        
        동작 과정:
            1. 코드 블록에서 다양한 패턴을 찾아 스킬을 추출
            2. 에러 메시지에서 특정 오류 유형을 찾아 스킬을 추출
            3. 텍스트에서 특정 키워드를 기반으로 스킬을 추출
            4. 분석 유형을 기반으로 특정 스킬을 추가
            5. 중복 스킬은 한 번만 추가되도록 처리 (add_skill 함수)
        """
        skills = []

        # 코드 분석을 위한 패턴
        code_patterns = {
            'Value': r'\b\d+(\.\d+)?\b',
            'Variable Assign': r'\b\w+\s*=\s*.+',
            'Operators': r'\b\w+\s*[+\-*/%]\s*\w+|\b\d+\s*[+\-*/%]\s*\d+',
            'Operands': r'\b\w+\s*[+\-*/%]\s*\w+|\b\d+\s*[+\-*/%]\s*\d+',
            'Type Convertor': r'\b(int|float|str)\b',
            'input function': r'\binput\(.+\)',
            'print function': r'\bprint\(.+\)',
            'Boolean Values': r'\b(True|False)\b',
            'Boolean Expressions': r'\b==\b|\b<=\b|\b>=\b|\b>\b|\b<\b',
            'If-Else Statements': r'\bif\b',
            'For Loops': r'\bfor\b',
            'While Loops': r'\bwhile\b',
            'Break Statement': r'\bbreak\b',
            'Continue Statement': r'\bcontinue\b',
            'return Statement': r'\breturn\b',
            'Local, Global Scope': r'\b(global|local)\b',
            'Lists': r'\[.*\]|\.append\(|\.insert\(|\.count\(|\.extend\(|\.index\(|\.reverse\(|\.sort\(|\.remove\(|\s*\+\s*\[.*\]|\s*\*\s*\d+|\[.*:.*\]|\.copy\(',
            'Dictionaries': r'\.keys\(|\.values\(|\.items\(|\.setdefault\(|\.copy\(',
            'Strings': r'\bstr\b',
            'Indexing': r'\[.*\]',
            'Import Statement': r'\bimport\b',
            'random': r'\brandom\b',
        }


        # 오류 메시지 분석을 위한 패턴
        error_patterns = {
            'SyntaxError': r'SyntaxError',
            'NameError': r'NameError',
            'TypeError': r'TypeError',
            'IndentationError': r'IndentationError',
            'ValueError': r'ValueError',
            'AttributeError': r'AttributeError',
            'IndexError': r'IndexError',
            'KeyError': r'KeyError',
            'TabError': r'TabError',
            'UnicodeDecodeError': r'UnicodeDecodeError',
            'FileNotFoundError': r'FileNotFoundError',
            'ModuleNotFoundError': r'ModuleNotFoundError',
            'ZeroDivisionError': r'ZeroDivisionError',
            'UnboundLocalError': r'UnboundLocalError',
            'ImportError': r'ImportError',
            'UnicodeEncodeError': r'UnicodeEncodeError',
            'LookupError': r'LookupError',
            'ConnectionError': r'ConnectionError',
            'RuntimeError': r'RuntimeError',
        }

        # 텍스트 분석을 위한 패턴
        text_patterns = {
            'Value': r'\b숫자\b|\b소수\b|\b정수\b|\b부동 소수\b',
            'Variable Assign': r'\b변수\b|\b할당\b|\b넣\b|\b넣을\b',
            'Operators': r'[\+\-\*/%]|\b더하기\b|\b빼기\b|\b곱하기\b|\b나누기\b|\b제곱\b',
            'Operands': r'\b\d+\s*[+\-*/%]\s*\d+\b|\b넣\b|\b빼\b',
            'Type Convertor': r'\b정수 변환\b|\b부동 소수\b|\b문자열\b|\btype\b',
            'input function': r'\b입력\b|\binput\b',
            'print function': r'\b출력\b|\b프린트\b|\bprint\b',
            'Boolean Values': r'\b참\b|\b거짓\b|\btrue\b|\bfalse\b',
            'Boolean Expressions': r'\b같음\b|\b같은\b|\b다름\b|\b다른\b|\b크다\b|\b큰\b|\b작다\b|\b작은\b|\b크거나 같다\b|\b크거나 같은\b|\b작거나 같다\b|\b작거나 같은\b',
            'Logical Operators': r'\b그리고\b|\b또는\b|\b부정\b|\b아닐 때\b',
            'If-Else Statements': r'\bif\b|\belse\b',
            'For Loops': r'\bfor\b|\bfor 루프\b|\bfor 문\b',
            'While Loops': r'\bwhile\b|\bwhile 조건문\b',
            'Break Statement': r'\b중단\b|\bbreak\b',
            'Continue Statement': r'\b계속\b|\bcontinue\b',
            'return Statement': r'\b리턴\b|\breturn\b',
            'Local, Global Scope': r'\b전역\b|\b지역\b',
            'Strings': r'\b문자열\b|\b스트링\b',
            'Indexing': r'\[.*\]|\b인덱스\b|\b인덱싱\b',
            'Lists': r'\b리스트\b',
            'Dictionaries': r'\b딕셔너리\b',
            'Import Statement': r'\bimport\b|\b수입\b',
            'random': r'\b랜덤\b',
            'File Operations': r'\b파일\b',
        }

        # 분석 텍스트 기반 스킬 패턴
        analysis_patterns = {
            "Explanation Skill": r'Explanation needed',
            "Debugging Skill": r'Bug fixing',
            "Translation Skill": r'Translation',
            "Refactoring Skill": r'Refactoring',
            "Time Complexity Analysis": r'Time complexity',
            "Intent Understanding": r'Intent understanding'
        }


        # 코드를 통한 스킬 추출
        for contents in codes:
            contents = str(contents)
            for skill, pattern in code_patterns.items():
                if re.search(pattern, contents):
                    self._add_skill(skills, skill)
                

        # 오류를 통한 스킬 추출
        for contents in errors:
            contents = str(contents)
            for skill, pattern in error_patterns.items():
                if re.search(pattern, contents):
                    self._add_skill(skills, skill)

        # 텍스트를 통한 스킬 추출
        for text in texts:
            text = str(text)
            for skill, pattern in text_patterns.items():
                if re.search(pattern, text):
                    self._add_skill(skills, skill)

        # 분석 텍스트를 통한 스킬 추출
        for item in analysis:
            item = str(item)
            for skill, pattern in analysis_patterns.items():
                if re.search(pattern, item):
                    self._add_skill(skills, skill)



        return skills


        

    def _embed_skill(self, skill_text):
        """
        주어진 스킬 텍스트를 임베딩하는 함수.

        Args:
            skill_text (str): 스킬을 설명하는 텍스트 문자열.

        Returns:
            스킬 텍스트에 대한 임베딩. Shape: [1, 512]

        """
        max_length = 512
        token_type = '[SKILL]'
        text = f"{token_type} {skill_text}" if skill_text else token_type

        tokens_ids = self.text_tokenizer.encode(text, add_special_tokens=False, max_length=max_length-2, truncation=True)
        input_ids = torch.tensor([self.text_tokenizer.cls_token_id] + tokens_ids + [self.text_tokenizer.sep_token_id]).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.text_model(input_ids)
            embeddings = outputs.last_hidden_state.max(dim=1).values

        return self.projection_text(embeddings)
    
    def _embed_next_skills(self, ex_id):
        """
        주어진 ex_id의 다음 문제를 풀기 위해 필요한 스킬들을 임베딩하는 함수

        Args:
            ex_id (int): 특정 문제 id

        Returns:
            추출된 스킬 텍스트에 대한 임베딩. Shape: [1, 512]
        """
        max_length = 512
        token_type = '[TARGET SKILL]'
        
        # ex_id에 해당하는 행 로드
        row = self.text_df.loc[self.text_df['exercise_id'] == ex_id].iloc[0]
        
        instruction_content = row['Instruction Content']
        solution_content = row['Solution Content']
        
        instruction_content = self._parse_content_column(instruction_content)
        solution_content = self._parse_content_column(solution_content)
        
        #instruction_content = 글, solution_content = 코드 포함
        skills = self._extract_skills(solution_content, [], instruction_content, [])
        
        if self.debug:
            logging.info(f"Extracted skills from solution: {skills}")
            logging.info(f"Instruction: {instruction_content}")
            logging.info(f"Solution: {solution_content}")
        
        skill_text = ' '.join(skills)

        text = token_type + ' ' + skill_text
        
        tokens_ids = self.text_tokenizer.encode(text, add_special_tokens=False, max_length=max_length-2, truncation=True)
        input_ids = torch.tensor([self.text_tokenizer.cls_token_id] + tokens_ids + [self.text_tokenizer.sep_token_id]).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.text_model(input_ids)
            embeddings = outputs.last_hidden_state.max(dim=1).values
        
        return self.projection_text(embeddings)

    
    def _create_user_problem_map(self):
        """
        {
            '사용자ID1': {
                문제ID1: [제출인덱스1, 제출인덱스2, ...],
                문제ID2: [제출인덱스1, 제출인덱스2, ...],
                ...
            },
            '사용자ID2': {
                문제ID1: [제출인덱스1, 제출인덱스2, ...],
                문제ID2: [제출인덱스1, 제출인덱스2, ...],
                ...
            },
            ...
        }

        실제 데이터
        {
            'da13b1835a64ace869a29a0ebc54f43befe8a64961d37798cb4f99ec1b183a6b': defaultdict(<class 'list'>, 
            {
                1551714: [7567, 7568, 7569, 7570, 7571, 7572], 
                1554364: [7573, 7574]
            })
            ...

            '431553096cc8235df17bf3edd61d57dfab41ed6d9c6b6025180f303124186d0b': defaultdict(<class 'list'>, 
            {
                1548385: [8378, 8379, 8380], 
                1548415: [8381], 
                1548498: [8382, 8383, 8384], 
                1548709: [8385, 8386, 8387, 8388], 
                1548907: [8389, 8390, 8391, 8392, 8393, 8394, 8395], 
                1549215: [8396, 8397, 8398, 8399, 8400, 8401, 8402, 8403, 8404, 8405, 8406, 8407, 8408, 8409, 8410, 8411, 8412, 8413, 8414], 
                1549445: [8415, 8416], 
                1549659: [8417, 8418, 8419, 8420, 8421, 8422, 8423, 8424, 8425, 8426, 8427, 8428, 8429, 8430, 8431, 8432, 8433, 8434, 8435, 8436], 
                1549839: [8437, 8438, 8439, 8440, 8441, 8442, 8443, 8444, 8445, 8446, 8447, 8448, 8449], 
                1549976: [8450, 8451, 8452, 8453, 8454], 1550130: [8455, 8456, 8457, 8458, 8459], 
                1550975: [8460, 8461, 8462, 8463], 
                1551307: [8464, 8465], 
                1551465: [8466, 8467, 8468, 8469, 8470], 
                1551598: [8471, 8472, 8473, 8474, 8475, 8476], 
                1551714: [8477, 8478, 8479, 8480, 8481, 8482, 8483, 8484, 8485, 8486, 8487, 8488, 8489, 8490, 8491, 8492, 8493], 
                1551839: [8494], 
                1551953: [8495, 8496], 
                1552026: [8497, 8498, 8499, 8500, 8501, 8502], 
                1552935: [8503, 8504], 
                1553025: [8505, 8506]
            })
            ...
        }
        """

        #각 학생(x_user_id)이 푼 문제(exercise_id)별로 제출된 코드의 인덱스들을(실제 code_df의 인덱스) 저장하는 역할
        #이를 통해 각 학생의 문제 풀이 이력을 관리할 수 있으며, 나중에 각 학생의 제출 내역에 접근하여 필요한 정보를 빠르게 가져올 수 있음
        user_problem_map = defaultdict(lambda: defaultdict(list))
        for idx, row in self.code_df.iterrows():
            user_problem_map[row['x_user_id']][row['exercise_id']].append(idx)
            
        if self.debug:
            logging.info(f"\n\nAdding index {idx} for user {row['x_user_id']} and problem {row['exercise_id']}")
                
        return user_problem_map


    def _get_positive_sample(self, user_id, problem_id, current_idx):
        """
        주어진 사용자가 특정 문제에 대해 제출한 다른 제출물을 가져오는 함수
        같은 사용자의 같은 문제에 대해 현재 제출물 (current_idx)과 다른 제출물 중 하나를 선택

        Args:
            user_id (str): 사용자 
            problem_id (int): 현재 문제 id
            current_idx (int): 현재 제출물의 인덱스

        Returns:
            int: positive 샘플로 선택된 제출물의 인덱스
        """
        possible_positives = [i for i in self.user_problem_map[user_id][problem_id] if i != current_idx]
        positive_idx = np.random.choice(possible_positives) if possible_positives else current_idx
        return positive_idx

    def _get_negative_sample(self, user_id, problem_id, current_idx):
        """
        주어진 사용자가 특정 문제에 대해 제출한 것과 다른 문제나 아예 다른 사용자의 제출물을 가져오는 함수
        사용자가 푼 다른 문제나 다른 사용자의 제출물 중 하나를 negative 샘플로 선택

        Args:
            user_id (str): 사용자 
            problem_id (int): 문제 id
            current_idx (int): 현재 제출물의 인덱스

        Returns:
            int: negative 샘플로 선택된 제출물의 인덱스
        """
        possible_negatives = []
        for prob_id in self.user_problem_map[user_id]:
            if prob_id != problem_id:
                possible_negatives.extend(self.user_problem_map[user_id][prob_id])
        if not possible_negatives:
            possible_negatives = [i for i in range(len(self.code_df)) if i != current_idx and self.code_df.iloc[i]['x_user_id'] != user_id]
        negative_idx = np.random.choice(possible_negatives) if possible_negatives else current_idx
        return negative_idx


    def _get_negative_problem_id(self, user_id, problem_id):
        possible_negatives = [prob_id for prob_id in self.user_problem_map[user_id] if prob_id != problem_id]
        
        if possible_negatives:
            negative_problem_id = random.choice(possible_negatives)
        else:
            negative_problem_id = problem_id
        
        return negative_problem_id


    def _get_question_positive_sample(self, user_id, problem_id):

        questions = self.help_center_log[(self.help_center_log['x_user_id'] == user_id) & (self.help_center_log['exercise_id'] == problem_id)]
        if not questions.empty:
            question = questions.iloc[0]['content']
            return self._embed_question(question).cpu()

        return torch.zeros(1, 512).cpu()

    def _get_question_negative_sample(self, user_id, problem_id):

        questions = self.help_center_log[(self.help_center_log['x_user_id'] != user_id) & (self.help_center_log['exercise_id'] != problem_id)]
        if not questions.empty:
            question = questions.iloc[0]['content']
            return self._embed_question(question).cpu()

        return torch.zeros(1, 512).cpu()

    def _get_code_submissions(self, user_id, problem_id):
        # user_id와 problem_id에 해당하는 코드 제출 내역을 반환
        submissions = self.code_df[(self.code_df['x_user_id'] == user_id) & (self.code_df['exercise_id'] == problem_id)]
        return submissions['contents'].tolist()

    def _get_problem_text(self, problem_id):
        # problem_id에 해당하는 문제 설명 텍스트 반환
        problem = self.text_df[self.text_df['exercise_id'] == problem_id]
        return problem['Instruction Content'].iloc[0] if not problem.empty else ""

    def _get_question_text(self, user_id, problem_id):
        questions = self.help_center_log[(self.help_center_log['x_user_id'] == user_id) & 
                                         (self.help_center_log['exercise_id'] == problem_id)&
                                         self.help_center_log['is_student'] == True]
        return questions['content'].tolist() if not questions.empty else []
    
    def _get_next_problem_id(self, user_id, current_problem_idx):
        problem_list = list(self.user_problem_map[user_id].keys())
        if current_problem_idx < len(problem_list) - 1:
            return problem_list[current_problem_idx + 1]
        return None



def collate_fn(batch):
    max_len = max(len(item[0]) for item in batch) #가장 긴 시퀀스 길이
    embedding_dim = batch[0][0].size(1) #512

    # Initialize padded tensors
    padded_embeddings = torch.zeros((len(batch), max_len, embedding_dim), dtype=torch.float)
    padded_anchors = torch.zeros((len(batch), max_len, embedding_dim), dtype=torch.float)
    padded_positives = torch.zeros((len(batch), embedding_dim), dtype=torch.float)
    padded_negatives = torch.zeros((len(batch), embedding_dim), dtype=torch.float)
    padded_question_positives = torch.zeros((len(batch), embedding_dim), dtype=torch.float)
    padded_question_negatives = torch.zeros((len(batch), embedding_dim), dtype=torch.float)
    targets = torch.zeros((len(batch),), dtype=torch.float)
    original_data = []

    for i, (embeddings, target, anchor, positive, negative, question_positive, question_negative, codet5_input_ids, codet5_attention_mask, codet5_targets, data) in enumerate(batch):
        padded_embeddings[i, :len(embeddings)] = embeddings
        padded_anchors[i, :len(anchor)] = anchor
        padded_positives[i, :] = positive
        padded_negatives[i, :] = negative
        padded_question_positives[i, :] = question_positive
        padded_question_negatives[i, :] = question_negative
        targets[i] = target
        original_data.append(data)
        
    return padded_embeddings, targets, padded_anchors, padded_positives, padded_negatives, padded_question_positives, padded_question_negatives, codet5_input_ids, codet5_attention_mask, codet5_targets, original_data

        
def load_targets(target_file_path):
    """
        target_dict = {
            ('user1', 1001): 1,
            ('user1', 1002): 0,
            ('user2', 1001): 1
        }

        리턴 형식

        ('de832cc1dc9bfecd21edbead3cfd08f97cc418e0b2cf7abddf7a06672e86737a', 1549839.0): 1.0, 
        ('de832cc1dc9bfecd21edbead3cfd08f97cc418e0b2cf7abddf7a06672e86737a', 1549976.0): 1.0, 
        ('de832cc1dc9bfecd21edbead3cfd08f97cc418e0b2cf7abddf7a06672e86737a', 1550130.0): 1.0, 
        ('de832cc1dc9bfecd21edbead3cfd08f97cc418e0b2cf7abddf7a06672e86737a', 1550975.0): 1.0, 
        ('de832cc1dc9bfecd21edbead3cfd08f97cc418e0b2cf7abddf7a06672e86737a', 1551307.0): 0.0, 
        ('de832cc1dc9bfecd21edbead3cfd08f97cc418e0b2cf7abddf7a06672e86737a', 1551465.0): 1.0,
        ...

    """
    target_df = pd.read_csv(target_file_path)

    target_df = target_df.sort_values(by=['x_user_id', 'exercise_id'])

    target_df['next_qualified'] = target_df.groupby('x_user_id')['qualified'].shift(-1)

    target_df['key'] = list(zip(target_df['x_user_id'], target_df['exercise_id']))

    target_df = target_df.dropna(subset=['next_qualified'])

    target_dict = dict(zip(target_df['key'], target_df['next_qualified']))

    return target_dict



def load_and_sample_dataset(config, question_model, device, cross):
    """
        데이터셋을 로드하고 훈련, 검증, 테스트 데이터셋으로 나누는 함수

        Args:
            config (dict): 데이터셋 경로 및 관련 설정을 포함하는 딕셔너리
            question_model (torch.nn.Module): 질문 임베딩을 위한 모델 (예: Codet5)
            device (str): 'cuda' 또는 'cpu'
            cross (bool): 교차 도메인 실험 여부를 나타내는 flag

        Returns:
            train_dataset (Subset): 훈련 데이터셋
            val_dataset (Subset): 검증 데이터셋
            test_dataset (Subset): 테스트 데이터셋
    """
    target_file_path = config['targets']
    target_dict = load_targets(target_file_path)
    
    dataset = SkillDataset(
        config['exercises'],
        config['submissions'],
        config['questions'],
        target_dict,
        question_model,
        device=device,
        debug=False
    )
    
    if cross == False:
        indices = list(range(len(dataset)))
        train_indices, test_indices = train_test_split(indices, test_size=0.2, shuffle = False)
        val_indices, test_indices = train_test_split(test_indices, test_size=0.5, shuffle = False)
        train_dataset = Subset(dataset, train_indices)
        val_dataset = Subset(dataset, val_indices)
        test_dataset = Subset(dataset, test_indices)
        return train_dataset, val_dataset, test_dataset
    else:
        indices = list(range(len(dataset)))
        train_indices, test_indices = train_test_split(indices, test_size=0.2, shuffle = False)
        train_dataset = Subset(dataset, train_indices)
        test_dataset = Subset(dataset, test_indices)
        return train_dataset, test_dataset


def prepare_and_split_data_loaders(dataset_configs, question_model, batch_size=4, shuffle=False, device='cuda', cross=False, dataset_key=None):
    """
        dataset_configs을 사용하여 데이터 로더를 준비하고 분할
        
        Args:
            dataset_configs (딕셔너리): 데이터셋 경로와 샘플링 비율을 포함
            question_model: codet5 embedder
            batch_size (int): DataLoader의 배치 크기.
            shuffle (bool): 데이터를 셔플할지 여부.
            device (str): 데이터를 로드할 장치 ('cuda' 또는 'cpu').
            cross (bool): False = in-domain, True = cross-domain
            dataset_key (str or list): 데이터셋의 키 (하나 또는 여러 개 가능)
        
        Returns:
            train_loader: 학습 데이터로더
            val_loader: 벨리데이션 데이터로더
            test_loader: 테스트 데이터로더
    """
    
    # 여러 개의 dataset_key를 받을 수 있도록 처리
    if isinstance(dataset_key, str):
        dataset_key = [dataset_key]  # 단일 키일 경우 리스트로 변환
    
    train_datasets = []
    val_datasets = []
    test_datasets = []

    # dataset_key 리스트에 있는 각 키에 대해 데이터셋 로드
    if cross == False:
        for key in dataset_key:
            if key not in dataset_configs:
                raise ValueError(f"dataset_key '{key}'가 dataset_configs에 없습니다.")
            
            # 해당 키에 맞는 데이터셋 로드
            single_config = dataset_configs[key]
            train_dataset, val_dataset, test_dataset = load_and_sample_dataset(single_config, question_model, device, cross)
            
            train_datasets.append(train_dataset)
            val_datasets.append(val_dataset)
            test_datasets.append(test_dataset)
        
        # 여러 개의 데이터셋을 결합
        combined_train_dataset = ConcatDataset(train_datasets)
        combined_val_dataset = ConcatDataset(val_datasets)
        combined_test_dataset = ConcatDataset(test_datasets)
        
        
        train_loader = DataLoader(combined_train_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
        val_loader = DataLoader(combined_val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        test_loader = DataLoader(combined_test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

        return train_loader, val_loader, test_loader
    
    else:
        # cross-domain 처리
        last_key = dataset_key[-1]  # dataset_key 리스트의 마지막 항목을 test로 사용
        for key in dataset_key[:-1]:  # 마지막을 제외한 모든 항목을 train에 사용
            if key not in dataset_configs:
                raise ValueError(f"dataset_key '{key}'가 dataset_configs에 없습니다.")
            
            # 해당 키에 맞는 데이터셋 로드
            train_dataset, val_dataset = load_and_sample_dataset(dataset_configs[key], question_model, device, cross)
            print(f"Loaded train dataset '{key}' size: {len(train_dataset)}")  # 크기 출력
            train_datasets.append(train_dataset)
            val_datasets.append(val_dataset)

        # 마지막 항목은 테스트 데이터셋으로 로드
        test_dataset, val_dataset = load_and_sample_dataset(dataset_configs[last_key], question_model, device, cross)
        test_datasets.append(test_dataset)
        test_datasets.append(val_dataset)
        
        combined_train_dataset = ConcatDataset(train_datasets)
        combined_val_dataset = ConcatDataset(val_datasets)
        combined_test_dataset = ConcatDataset(test_datasets)


        train_loader = DataLoader(combined_train_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
        val_loader = DataLoader(combined_val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        test_loader = DataLoader(combined_test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        
        print(f"Train dataset size: {len(train_loader)}")
        print(f"Validation dataset size: {len(val_loader)}")
        print(f"Test dataset size: {len(test_dataset)}")
        
        return train_loader, val_loader, test_loader




    


def debug():

    question_tokenizer = RobertaTokenizer.from_pretrained("Salesforce/codet5p-220m")
    question_tokenizer.add_special_tokens({'additional_special_tokens': ['[SKILL]', '[TARGET SKILL]', '[QUESTION]', '[TARGET]', '[CODE]', '[TEXT]']})
    question_model = T5ForConditionalGeneration.from_pretrained("Salesforce/codet5p-220m")
    question_model.resize_token_embeddings(len(question_tokenizer))
    dataset_configs = {
        # exercises: 문제 정보 파일
        # submissions: 학생 문제 제출 파일
        # questions: 학생 질문 파일
        # targets: 타겟 파일
        # sampling_ratio: cross domain 실험 시 random sampling ratio
            "18873": {
                "exercises": "/home/doyounkim/Transformer/data/18873/18873_pre_dropone.csv",
                "submissions": "/home/doyounkim/Transformer/data/18873/18873_exercises_skill.csv",
                "questions": "/home/doyounkim/Transformer/data/18873/18873_helpcenter_log_skill.csv",
                "targets": "/home/doyounkim/Transformer/data/18873/18873_final_scores.csv",
                "sampling_ratio": 0.01
            },
            "18818": {
                "exercises": "/home/doyounkim/Transformer/data/18818/18818_pre_dropone.csv",
                "submissions": "/home/doyounkim/Transformer/data/18818/18818_exercises_skill.csv",
                "questions": "/home/doyounkim/Transformer/data/18818/18818_helpcenter_log_skill.csv",
                "targets": "/home/doyounkim/Transformer/data/18818/18818_final_scores.csv",
                "sampling_ratio": 1
            },
            "18945": {
                "exercises": "/home/doyounkim/Transformer/data/18945/18945_pre_dropone.csv",
                "submissions": "/home/doyounkim/Transformer/data/18945/18945_exercises_skill.csv",
                "questions": "/home/doyounkim/Transformer/data/18945/18945_helpcenter_log_skill.csv",
                "targets": "/home/doyounkim/Transformer/data/18945/18945_final_scores.csv",
                "sampling_ratio": 0.1
            }, 
            "18888": {
                "exercises": "/home/doyounkim/Transformer/data/18888/18888_pre_dropone.csv",
                "submissions": "/home/doyounkim/Transformer/data/18888/18888_exercises_skill.csv",
                "questions": "/home/doyounkim/Transformer/data/18888/18888_helpcenter_log_skill.csv",
                "targets": "/home/doyounkim/Transformer/data/18888/18888_final_scores.csv",
                "sampling_ratio": 1
            }
        }
    
    train_loader, val_loader, test_loader = prepare_and_split_data_loaders(
        dataset_configs,
        question_model, 
        batch_size=1,
        shuffle=False,
        device='cuda',
        cross=False,  
        # dataset_key=['18818', '18945', '18873', '18888'] 
        dataset_key=['18818'] 
        # dataset_key = ['18945', '18873']
    )


    i = 0
    for batch_idx, (embeddings, targets, anchor_embeddings, positive_embeddings, negative_embeddings, question_positive_emb, question_negative_emb, codet5_input_ids, codet5_attention_mask, codet5_targets
        ) in enumerate(train_loader):
            print(f"Batch {batch_idx + 1}:")
            print(f"Embeddings Shape: {embeddings.shape}")
            # print(f"Embeddings: {embeddings}")
            print(f"inputs type: {embeddings.dtype}")
            print(f"Targets: {targets}")
            print(f"Targets shape: {targets.shape}")
            print(f"Anchor Embeddings Shape: {anchor_embeddings.shape}")
            print(f"Anchor Embeddings Type: {anchor_embeddings.dtype}")
            print(f"Positive Embeddings Shape: {positive_embeddings.shape}")
            print(f"Negative Embeddings Shape: {negative_embeddings.shape}")
            print(f"Question Positive Embeddings Shape: {question_positive_emb.shape}")
            print(f"Question Negative Embeddings Shape: {question_negative_emb.shape}")
            print(f"codet5_input_ids: {codet5_input_ids.shape}")
            print(f"codet5_input_ids type: {codet5_input_ids.dtype}")
            print(f"codet5_attention_mask: {codet5_attention_mask.shape}")
            print(f"codet5_targets: {codet5_targets.shape}")
            codet5_loss = question_model(input_ids=codet5_input_ids.long(), attention_mask=codet5_attention_mask, labels=codet5_targets.long()).loss
            print(f"Codet5 Loss: {codet5_loss}")
            print(f"Codet5 Loss shape: {codet5_loss.shape}")
            print(f"Codet5 Loss type: {codet5_loss.dtype}")    
            print()
            logging.info(f"\n\n=========================================================")
            logging.info(f"Batch {batch_idx + 1}:")
            logging.info(f"Embeddings Shape: {embeddings.shape}")
            logging.info(f"Targets: {targets}")
            logging.info(f"Anchor Embeddings Shape: {anchor_embeddings.shape}")
            logging.info(f"Positive Embeddings Shape: {positive_embeddings.shape}")
            logging.info(f"Negative Embeddings Shape: {negative_embeddings.shape}")
            logging.info(f"Question Positive Embeddings Shape: {question_positive_emb.shape}")
            logging.info(f"Question Negative Embeddings Shape: {question_negative_emb.shape}")
            logging.info(f"codet5_input_ids: {codet5_input_ids.shape}")
            logging.info(f"codet5_attention_mask: {codet5_attention_mask.shape}")
            logging.info(f"codet5_targets: {codet5_targets.shape}")
            codet5_loss = question_model(input_ids=codet5_input_ids.long(), attention_mask=codet5_attention_mask, labels=codet5_targets.long()).loss
            logging.info(f"Codet5 Loss: {codet5_loss}")
            logging.info(f"Codet5 Loss: {codet5_loss}")
            logging.info(f"Codet5 Loss shape: {codet5_loss.shape}")
            logging.info(f"Codet5 Loss type: {codet5_loss.dtype}") 
            logging.info(f"\n\n=========================================================")
            print()
            i += 1
            if i == 100:
                break

if __name__ == "__main__":
    debug()
