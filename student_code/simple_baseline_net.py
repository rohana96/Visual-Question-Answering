import sys

import torch
import torch.nn as nn

sys.path.append('../')
from VQA.external.googlenet.googlenet import googlenet


class SimpleBaselineNet(nn.Module):
    """
    Predicts an answer to a question about an image using the Simple Baseline for Visual Question Answering (Zhou et al, 2017) paper.
    """

    def __init__(self, word_in_dim=5747, img_feature_dim=1024, word_feature_dim=1024, out_dim=5217):  # 2.2 TODO: add arguments needed
        super().__init__()
        # ----------------- 2.2 TODO

        model = googlenet(pretrained=True).eval()  ## TODO
        self.image_feature_extractor = googlenet(pretrained=True)
        self.image_feature_extractor.requires_grad = False
        self.word_feature_extractor = nn.Linear(word_in_dim, word_feature_dim)
        self.classifier = nn.Linear(word_feature_dim + img_feature_dim, out_dim)
        # self.softmax = nn.Softmax()
        # -----------------

    def forward(self, image, question_encoding=None):
        # ----------------- 2.2 TODO
        image.cuda()
        question_encoding.cuda()
        image_feature = self.image_feature_extractor(image)[-1]

        question_encoding = question_encoding.amax(dim=1)

        word_feature = self.word_feature_extractor(question_encoding)
        comb_feature = torch.cat([image_feature, word_feature], dim=-1)
        out = self.classifier(comb_feature)
        # out = self.softmax(out)
        return out
        # -----------------


def test():
    img = torch.randn(10, 3, 224, 224)
    quest = torch.randn(10, 5747)
    model = SimpleBaselineNet()
    out = model(img, quest)
    print(out.shape)


if __name__ == "__main__":
    test()
