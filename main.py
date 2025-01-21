import sys
sys.path.append('/home/doyounkim/sqkt')
import random
import torch
from torch import nn
from transformers import T5ForConditionalGeneration, RobertaTokenizer
from layers.multi_head_attention import MultiHeadAttention
from embedding.transformer_embedding import TransformerEmbedding
from blocks.encoder_layer import EncoderLayer
from data.data_loader import prepare_and_split_data_loaders
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
import torch.nn.functional as F
import argparse, os
from collections import defaultdict
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
import numpy as np


def get_config():
    parser = argparse.ArgumentParser(description='Knowledge Tracing Transformer Configuration')
    
    # Model parameters
    parser.add_argument('--d_model', type=int, default=512, help='Dimension of the model')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=6, help='Number of transformer layers')
    parser.add_argument('--ffn_hidden', type=int, default=3072, help='Hidden size of the feedforward network')
    parser.add_argument('--dropout_rate', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--max_len', type=int, default=1024, help='Maximum sequence length')

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=3e-5, help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--cross', type=bool, default=False, help='If train is in-domain or cross-domain')

    # Other settings
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use for training')
    parser.add_argument('--tokenizer_name', type=str, default='Salesforce/codet5-small', help='Name of the tokenizer to use')

    args = parser.parse_args()

    tokenizer = RobertaTokenizer.from_pretrained(args.tokenizer_name)
    tokenizer.add_special_tokens({'additional_special_tokens': ['[SKILL]', '[TARGET SKILL]', '[QUESTION]', '[TARGET]', '[CODE]', '[TEXT]']})
    
    args.vocab_size = len(tokenizer)

    return args, tokenizer

class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        distance_positive = F.pairwise_distance(anchor, positive, p=2)
        distance_negative = F.pairwise_distance(anchor, negative, p=2)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean()

class CustomLoss(nn.Module):
    def __init__(self, margin=1.0, tokenizer=None):
        super(CustomLoss, self).__init__()
        self.margin = margin
        self.triplet_loss = TripletLoss(margin)
        self.prediction_loss = nn.BCELoss()
        self.codet5_loss = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id if tokenizer else -100) #패딩토큰 무시

    def forward(self, outputs, targets, anchor_out, positive_out, negative_out, question_positive_out, question_negative_out, codet5_logits, codet5_targets):
        """
        all data are torch.float32


        outputs = 모델 예측값 (batch size, 1)
        targets = 타겟 (batch size, 1)
        anchor_out = anchor (batch size, # of submission, 512)
        positive_out = positive sample (변수 이름 바꾸기) (batch size, 1, 512)
        negative_out = negative sample (batch size, 1, 512)
        question_positive_out = question positive sample (batch size, 1, 512)
        question_negative_out = question_negative_sample (batch size, 1, 512)
        codet5_loss = codet5 finetuning loss (training 함수에서 계산) (스칼라 텐서)
        """

        if len(anchor_out.shape) == 3 and len(positive_out.shape) == 2:
            positive_out = positive_out.unsqueeze(1).expand(-1, anchor_out.size(1), -1)
        if len(anchor_out.shape) == 3 and len(negative_out.shape) == 2:
            negative_out = negative_out.unsqueeze(1).expand(-1, anchor_out.size(1), -1)

        if len(anchor_out.shape) == 3 and len(question_positive_out.shape) == 2:
            question_positive_out = question_positive_out.unsqueeze(1).expand(-1, anchor_out.size(1), -1)
        if len(anchor_out.shape) == 3 and len(question_negative_out.shape) == 2:
            question_negative_out = question_negative_out.unsqueeze(1).expand(-1, anchor_out.size(1), -1)

        con_loss_code = self.triplet_loss(anchor_out, positive_out, negative_out)
        con_loss_question = self.triplet_loss(anchor_out, question_positive_out, question_negative_out)
        pred_loss = self.prediction_loss(outputs, targets)
        codet5_loss = self.codet5_loss(codet5_logits.view(-1, codet5_logits.size(-1)), codet5_targets.view(-1))

        
        total_loss = pred_loss + 0.5 * (con_loss_code + con_loss_question) + codet5_loss
        return total_loss

class KnowledgeTracingTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, ffn_hidden, drop_prob, max_len, device, question_model, tokenizer):
        super(KnowledgeTracingTransformer, self).__init__()
        self.embedding = TransformerEmbedding(vocab_size=vocab_size, d_model=d_model, max_len=max_len, drop_prob=drop_prob, device=device)
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model=d_model, ffn_hidden=ffn_hidden, n_head=num_heads, drop_prob=drop_prob) for _ in range(num_layers)])
        self.multihead_attention = MultiHeadAttention(d_model=d_model, n_head=num_heads)
        self.question_model = question_model

        self.output_layer = nn.Sequential(
            nn.Linear(512, 128), 
            nn.ReLU(),                
            nn.Dropout(0.5),          
            nn.Linear(128, 64),  
            nn.ReLU(),                
            nn.Dropout(0.5), 
            nn.Linear(64, 1), 
            nn.Sigmoid()
        )

        self.pad_token = tokenizer.pad_token_id
        self.device = device

    def forward(self, inputs, anchor, problem_text_positive_sample, problem_text_negative_sample, question_positive_sample, question_negative_sample, codet5_input_ids, codet5_attention_mask, codet5_targets):
        """
        all data are torch.float32

        inputs = 누적된 학생 문제 풀이 데이터 (batch size, n, 512) *n은 변동값
        anchor = anchor (batch size, # of submission, 512)
        problem_text_positive_sample = 문제 positive sample (변수 이름 바꾸기) (batch size, 1, 512)
        problem_text_negative_sample = 문제 negative sample (batch size, 1, 512)
        question_positive_sample = question positive sample (batch size, 1, 512)
        question_negative_sample = question_negative_sample (batch size, 1, 512)
        codet5_input_ids = 학생 질문 토큰 (batch size, 1, 512)
        codet5_attention_masks = 학생 질문 어텐션 마스크 토큰 (batch size, 1, 512)
        codet5_targets = 교사 응답 토큰 (batch size, 1, 512)
        """ 
        outputs = self.process_embedding(inputs) #transformer encoder 통과한 결과물 [batch size, seq_length, 512]

        ####### shape check done ##########
        # 따라서, dim=1을 사용해 시퀀스의 길이를 줄이고, batch_size x feature_size 형태의 출력을 얻어 이후 예측에 사용
        prediction = self.output_layer(torch.max(outputs, dim=1)[0]) # max pooling 후 predict layer 통과 #
        #torch.max(outputs, dim=1)[0] ([batch size, 512])
        #prediction shape: [batch size, 1]

        if codet5_input_ids is not None:
            codet5_input_ids = codet5_input_ids.long()
            codet5_targets = codet5_targets.long()

            # codet5 loss 계산
            codet5_logits = self.question_model(input_ids=codet5_input_ids, attention_mask=codet5_attention_mask, labels=codet5_targets).logits
            #codet5_loss shape: 스칼라텐서

            return prediction, codet5_logits, anchor, problem_text_positive_sample, problem_text_negative_sample, question_positive_sample, question_negative_sample, codet5_targets #train
        else:
            return prediction, anchor, problem_text_positive_sample, problem_text_negative_sample, question_positive_sample, question_negative_sample #evaluation

    def process_embedding(self, inputs):
        """
        입력 시퀀스를 임베딩하고, transformer encoder layer를 통과시켜 출력을 생성하는 함수
        
        Args:
            inputs (Tensor): 입력 데이터, shape은 [batch_size, sequence_length, 512]이며, sequence_length가 매우 클 수 있음

        Returns:
            Tensor: transformer encoder layer를 통과한 최종 임베딩된 입력, shape은 [batch_size, max_sequence_length, embedding_dim]
        """ 
        max_sequence_length = 1024
        inputs = inputs = inputs[:, -max_sequence_length:, :] 
        src_mask = self.create_padding_mask(inputs)

        embedded_input = self.embedding(inputs) #positional encoding
        for layer in self.encoder_layers:
            embedded_input = layer(embedded_input, src_mask) #transformer encoder layer
        return embedded_input

    
    def create_padding_mask(self, seq):
        """
        seq는 [batch_size, sequence_length, hidden_size] 형태.
        패딩 여부를 판단하기 위해 hidden_size를 무시하고 sequence_length만 사용해야 함.
        여기서는 seq의 첫 번째 hidden dimension만을 사용하여 패딩 마스크를 생성.

        1. seq.sum(dim=-1)
        hidden_size 차원을 따라 합계를 계산
        [batch_size, sequence_length, hidden_size] -> [batch_size, sequence_length]
        패딩된 위치는 모든 값이 0이므로 합계도 0

        2. (seq.sum(dim=-1) != 0)
        합계가 0이 아닌 위치는 True (실제 데이터)
        합계가 0인 위치는 False (패딩)
        [batch_size, sequence_length] 형태의 불리언 마스크

        3. unsqueeze(1).unsqueeze(2)
        attention 계산을 위해 차원 추가
        [batch_size, sequence_length] -> [batch_size, 1, 1, sequence_length]
        """
        mask = (seq.sum(dim=-1) != 0).unsqueeze(1).unsqueeze(2)
        return mask

def calculate_metrics(outputs, labels):
    """
    모델의 예측 결과에 대한 성능 계산 함수

    Args:
        outputs (list or tensor): 모델의 예측 결과 (확률값)
        labels (list or tensor): 실제 정답 라벨 (0 또는 1로 제공)

    Returns:
        tuple: accuracy, precision, recall, f1, auc 점수들의 튜플
    """
    outputs = torch.tensor(outputs).numpy()
    labels = torch.tensor(labels).numpy()
    labels = labels.astype(int)

    binary_outputs = (outputs > 0.5).astype(int)

    accuracy = accuracy_score(labels, binary_outputs)
    precision = precision_score(labels, binary_outputs)
    recall = recall_score(labels, binary_outputs)
    f1 = f1_score(labels, binary_outputs)
    auc = roc_auc_score(labels, outputs)

    return accuracy, precision, recall, f1, auc


def train_model(model, train_loader, val_loader, optimizer, custom_loss, device, num_epochs):
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        num_batches = 0

        for data in tqdm(train_loader, desc=f'Epoch {epoch+1} Training', leave=False):
            padded_embeddings, targets, anchors, problem_text_positives, problem_text_negatives, question_positives, question_negatives, codet5_input_ids, codet5_attention_mask, codet5_targets, original_data = data
            padded_embeddings, targets, anchors, problem_text_positives, problem_text_negatives, question_positives, question_negatives = padded_embeddings.to(device), targets.to(device), anchors.to(device), problem_text_positives.to(device), problem_text_negatives.to(device), question_positives.to(device), question_negatives.to(device)
            codet5_input_ids, codet5_attention_mask, codet5_targets= codet5_input_ids.to(device), codet5_attention_mask.to(device), codet5_targets.to(device)
            
            optimizer.zero_grad()
            prediction, codet5_logits, anchor_out, positive_out, negative_out, question_positive_out, question_negative_out, codet5_targets = model(padded_embeddings, anchors, problem_text_positives, problem_text_negatives, question_positives, question_negatives, codet5_input_ids, codet5_attention_mask, codet5_targets)
            loss = custom_loss(
                prediction, targets.unsqueeze(1), anchor_out, positive_out, negative_out, 
                question_positive_out, question_negative_out, codet5_logits, codet5_targets

            )
            loss.backward(retain_graph = True)
            optimizer.step()
            total_train_loss += loss.item()
            num_batches += 1

        
        print(f'Epoch {epoch+1}/{num_epochs} completed, Average Train Loss: {total_train_loss / len(train_loader)}')
        
        criterion = nn.BCELoss() 
        all_val_outputs, all_val_targets, _, _ = evaluate_model(model, val_loader, criterion, device)
        accuracy, precision, recall, f1, auc = calculate_metrics(all_val_outputs, all_val_targets)
        print(f'Validation Metrics - Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}, AUC: {auc}')

        torch.cuda.empty_cache()


def evaluate_model(model, test_loader, criterion, device):
    model.eval()
    total_test_loss = 0
    all_outputs = []
    all_targets = []
    all_question_data = []

    for data in tqdm(test_loader, desc='Final Validation'):
        inputs, targets, _, _, _, _, _, _, _, _, original_data = data
        inputs, targets = inputs.to(device), targets.to(device)
        with torch.no_grad():
            outputs, _, _, _, _, _ = model(inputs, None, None, None, None, None, None, None, None)
            targets = targets.unsqueeze(1)
            loss = criterion(outputs, targets.float())
            total_test_loss += loss.item()
            all_outputs.append(outputs.cpu())
            all_targets.append(targets.cpu())

            all_question_data.extend(original_data)

    all_outputs = torch.cat(all_outputs).numpy()
    all_targets = torch.cat(all_targets).numpy()
    return all_outputs, all_targets, total_test_loss / len(test_loader), all_question_data



def main():
    config, tokenizer = get_config()
    """
    # transformer parameters
    d_model: Model embedding dimension size. This is the size of the hidden states in the transformer.
    num_heads: Number of attention heads in the multi-head attention mechanism. Each attention head computes attention independently and the results are concatenated.
    num_layers: Number of transformer encoder layers. This defines the depth of the model.
    ffn_hidden: The hidden size of the feedforward neural network in each transformer encoder layer. Typically larger than d_model for a deeper transformation.
    dropout_rate: Dropout rate for regularization. It prevents overfitting by randomly setting a fraction of input units to 0 during training.
    max_len: Maximum sequence length that the transformer can handle. Inputs longer than this will be truncated.

    # training parameters
    batch_size: Batch size for training. 
    learning_rate: Learning rate for the optimizer. This controls how much to change the model in response to the estimated error each time the model weights are updated.
    num_epochs: Number of epochs for training. 
    cross: Indicates if cross-domain training is applied. When set to True, cross-domain data is used for training.

    # other settings
    device: Device to use for training (either "cpu" or "cuda" for GPU training).
    tokenizer_name: default='Salesforce/codet5-small'
    """

    # 모델 파라미터
    vocab_size = config.vocab_size
    d_model = config.d_model #model embedding dimension size (default = 512)
    num_heads = config.num_heads #number of attention heads in multi-head attention (default = 8)
    num_layers = config.num_layers #number of transformer encoder layers (default = 6)
    ffn_hidden = config.ffn_hidden #hidden size of feed forward network (default = 3072)
    dropout_rate = config.dropout_rate #drop out rate of transformer encoder regularization (default = 0.1)
    max_len = config.max_len #maximum sequence length that transformer encoder handle (default = 1024)
    ########################

    # 학습 파라미터
    device = config.device
    learning_rate = config.learning_rate
    num_epochs = config.num_epochs
    batch_size = config.batch_size
    cross = config.cross

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
        batch_size=batch_size,
        shuffle=False,
        device=device,
        cross=cross,  
        # dataset_key=['18818', '18945', '18873', '18888'] 
        dataset_key=['18818'] 
        # dataset_key = ['18945', '18873']
    )


    model = KnowledgeTracingTransformer(vocab_size, d_model, num_heads, num_layers, ffn_hidden, dropout_rate, max_len, device, question_model, question_tokenizer)
    model.to(device)

    # 손실 함수와 옵티마이저 설정
    custom_loss = CustomLoss(margin=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCELoss() 

    
    train_model(model, train_loader, val_loader, optimizer, custom_loss, device, num_epochs)
        

    all_test_outputs, all_test_targets, avg_test_loss, all_question_data = evaluate_model(model, test_loader, criterion, device)
    test_accuracy, test_precision, test_recall, test_f1, test_auc = calculate_metrics(all_test_outputs, all_test_targets)

    print(f'Final Test Results - Loss: {avg_test_loss}, AUC: {test_auc}, Accuracy: {test_accuracy}, Precision: {test_precision}, Recall: {test_recall}, F1 Score: {test_f1}')



if __name__ == "__main__":
    main()