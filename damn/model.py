import copy
import torch
import torch.nn as nn
import pytorch_lightning as pl


class Attention(nn.Module):
    def __init__(self, feature_dim: int, domain_num: int):
        super().__init__()
        self.W = nn.Parameter(torch.FloatTensor(feature_dim, domain_num).uniform_(-0.1, 0.1))


    def forward(self, x: torch.Tensor):
        x = x.view(x.size()[0],-1)
        attention_logit = torch.matmul(x, self.W)
        normalized_attention = nn.functional.softmax(attention_logit, dim=1)
        return attention_logit, normalized_attention


class DAMN(nn.Module):
    def __init__(self, backbone: nn.Module, head_network: nn.Module, feature_dim: int, domain_num: int):
        super().__init__()
        self.backbone = backbone
        self.heads = [copy.deepcopy(head_network) for i in range(domain_num)]
        self.attention = Attention(feature_dim, domain_num)

    
    def forward(self, x):
        feature = self.backbone(x)
        attention_logit, normalized_attention = self.attention(feature)
        predictions = torch.cat([head(feature).unsqueeze(1) for head in self.heads], dim=1)
        
        #TODO: invent something clearer
        while (len(normalized_attention.size()) != len(predictions.size())):
            normalized_attention = normalized_attention.unsqueeze(-1)
        
        weighted_prediction =  torch.sum(normalized_attention * predictions, dim=1)
        return weighted_prediction, predictions, attention_logit



class DAMNTrain(pl.LightningModule):
    def __init__(self, backbone: nn.Module, head:nn.Module, feature_dim: int, num_domain: int, consistency_coef:float=0.1, domain_coef:float=0.1, consistency_loss_function=torch.nn.MSELoss()):
        super().__init__()
        self.model = DAMN(backbone,head,feature_dim, num_domain)
        self.consistency_loss_function = consistency_loss_function
        self.consistency_coef = consistency_coef
        self.domain_coef = domain_coef
        self.domain_loss_function = nn.CrossEntropyLoss()


    def forward(self, x):
        final_pred, domain_predictions, attention_scores = self.model(x)
        return final_pred


    def training_step(self, batch, batch_idx):
        x, domain_label = batch
        final_pred, domain_predictions, attention_scores = self.model(x)
        final_pred_concat = final_pred.unsqueeze(1).expand_as(domain_predictions)
        consistency_loss = self.consistency_coef * self.consistency_loss_function(final_pred_concat, domain_predictions)
        domain_loss = self.domain_coef * self.domain_loss_function(attention_scores, domain_label)
        return consistency_loss + domain_loss, final_pred
