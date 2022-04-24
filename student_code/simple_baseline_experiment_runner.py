import sys

import torch

sys.path.append("./student_code")
from simple_baseline_net import SimpleBaselineNet
from experiment_runner_base import ExperimentRunnerBase
from vqa_dataset import VqaDataset
from torchvision.transforms import transforms
from torch.nn.utils import clip_grad_norm_
import sys
import torch.optim as optim

sys.path.append('..')


class SimpleBaselineExperimentRunner(ExperimentRunnerBase):
    """
    Sets up the Simple Baseline model for training. This class is specifically responsible for creating the model and optimizing it.
    """

    def __init__(self, train_image_dir, train_question_path, train_annotation_path,
                 test_image_dir, test_question_path, test_annotation_path, batch_size, num_epochs,
                 num_data_loader_workers, cache_location, lr, log_validation, exp_name='simple'):
        # ----------------- 2.3 TODO: set up transform
        # Resizing to fit network input size;
        # Normalize to [0, 1] and convert from (H, W, 3) to (3, H, W);
        # Subtract mean [0.485, 0.456, 0.406] and divide by standard deviation [0.229, 0.224, 0.225] computed from ImageNet for each channel.    
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        ])

        # -----------------
        train_dataset = VqaDataset(image_dir=train_image_dir,
                                   question_json_file_path=train_question_path,
                                   annotation_json_file_path=train_annotation_path,
                                   image_filename_pattern="COCO_train2014_{}.jpg",
                                   transform=transform,
                                   # ----------------- 2.4 TODO: fill in the arguments
                                   question_word_to_id_map=None,
                                   answer_to_id_map=None,
                                   answer_word_list=None,
                                   id_to_answer_map=None
                                   # -----------------
                                   )
        val_dataset = VqaDataset(image_dir=test_image_dir,
                                 question_json_file_path=test_question_path,
                                 annotation_json_file_path=test_annotation_path,
                                 image_filename_pattern="COCO_val2014_{}.jpg",
                                 transform=transform,
                                 # ----------------- 2.4 TODO: fill in the arguments
                                 question_word_to_id_map=train_dataset.question_word_to_id_map,
                                 answer_to_id_map=train_dataset.answer_to_id_map,
                                 answer_word_list=train_dataset.answer_word_list,
                                 id_to_answer_map=train_dataset.id_to_answer_map
                                 # -----------------
                                 )

        model = SimpleBaselineNet()
        exp_name = exp_name

        super().__init__(train_dataset, val_dataset, model, batch_size, num_epochs, num_data_loader_workers, exp_name=exp_name)

        # ----------------- 2.5 TODO: set up optimizer
        self.optimizer = optim.SGD([{'params': model.word_feature_extractor.parameters(), 'lr': 0.8},
                                    {'params': model.classifier.parameters(), 'lr': 0.01}],
                                   lr=1e-3)
        # -----------------
        self.criterion = torch.nn.CrossEntropyLoss()

    def _optimize(self, predicted_answers, true_answer_ids):
        # ----------------- 2.7 TODO: compute the loss, run back propagation, take optimization step.
        true_answer_ids = torch.mean(true_answer_ids, dim=1)
        loss = self.criterion(predicted_answers, true_answer_ids)

        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self._model.parameters(), 20)
        self.optimizer.step()

        self._model.classifier.weight.data.clamp_(-20, 20)
        self._model.word_feature_extractor.weight.data.clamp_(-1500, 1500)
        return loss
        # -----------------
