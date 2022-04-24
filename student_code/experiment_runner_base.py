import os

import torch
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm

USE_WANDB = True
step_train = 0
step_val = 0
from PIL import Image
import torchvision
import torch.nn.functional as F
from torchvision.transforms import transforms
from torch.utils.tensorboard import SummaryWriter


class ExperimentRunnerBase(object):
    """
    This base class contains the simple train and validation loops for your VQA experiments.
    Anything specific to a particular experiment (Simple or Coattention) should go in the corresponding subclass.
    """

    def __init__(self, train_dataset, val_dataset, model, batch_size, num_epochs, num_data_loader_workers=4, log_validation=True, exp_name=''):
        self._model = model
        self._num_epochs = num_epochs
        self._log_freq = 10  # Steps
        self._test_freq = 1  # 250  # Steps
        self.exp_name = exp_name
        self.val_dataset = val_dataset
        self._train_dataset_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_data_loader_workers)

        # If you want to, you can shuffle the validation dataset and only use a subset of it to speed up debugging
        self._val_dataset_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_data_loader_workers)

        # Use the GPU if it's available.
        self._cuda = torch.cuda.is_available()

        if self._cuda:
            self._model = self._model.cuda()

        self._log_validation = log_validation

        ## wandb
        if USE_WANDB:
            wandb.init(
                project="vlr_hw4",
                name=f"{self.exp_name}"
            )
            self.text_table = wandb.Table(columns=["epoch", "val_step", "question", "image", "pred_answer", "gt_answer", "sample_idx"])

        ## tensorboard
        self.writer = SummaryWriter('runs/' + self.exp_name)

        self.val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    ## tensorboard
    def _collect_data_for_visualization(self, batch_data, pred_idx, gt_idx, num_samples=10):
        if torch.is_tensor(pred_idx):
            pred_idx = pred_idx.tolist()
        if torch.is_tensor(gt_idx):
            gt_idx = gt_idx.tolist()

        images = []
        questions, pred_answer, gt_answer = "", "", ""
        dataset = self._val_dataset_loader.dataset

        for i in range(num_samples):
            idx = batch_data['idx'][i]
            fname = dataset.index_list[idx]
            qid = int(fname)
            image_id = dataset._vqa.qa[qid]['image_id']
            image_id = '{:0>12}'.format(image_id)

            image_path = os.path.join(
                dataset._image_dir, dataset._image_filename_pattern.format(image_id))
            image_PIL = Image.open(image_path).convert('RGB')

            image = self.val_transform(image_PIL)
            images.append(image)

            questions += dataset._vqa.qqa[qid]['question'] + '\n\n'
            pred_answer += dataset.id_to_answer_map[pred_idx[i]] + '\n\n'
            gt_answer += dataset.id_to_answer_map[gt_idx[i]] + '\n\n'

        return images, questions, pred_answer, gt_answer

    ## wandb
    def _visualize(self, batch_data, idx, pred_idx, gt_idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        # idx = batch_data['idx'][idx]
        fname = self.val_dataset.index_list[idx]
        qid = int(fname)
        image_id = self.val_dataset._vqa.qa[qid]['image_id']
        image_id = '{:0>12}'.format(image_id)

        # image
        image_path = os.path.join(
            self.val_dataset._image_dir, self.val_dataset._image_filename_pattern.format(image_id))
        image = Image.open(image_path).convert('RGB')

        # question
        question = self.val_dataset._vqa.qqa[qid]['question']

        # gt_answer
        gt_answer = self.val_dataset.id_to_answer_map[gt_idx]

        # pred_answer
        pred_answer = self.val_dataset.id_to_answer_map[pred_idx]

        return image, question, pred_answer, gt_answer

    def _optimize(self, predicted_answers, true_answers):
        """
        This gets implemented in the subclasses. Don't implement this here.
        """
        raise NotImplementedError()

    def validate(self, epoch):
        # ----------------- 2.8 TODO
        # Should return your validation accuracy
        global step_val
        loss = 0.0
        correct, total = 0.0, 0.0
        # TODO fix accuracy calculation
        idxs, pred_idx, gt_idx = None, None, None
        print(f"running validation for epoch: {epoch}")

        num_batches = len(self._val_dataset_loader)
        for batch_id, batch_data in enumerate(self._val_dataset_loader):
            idxs = batch_data['idx']
            question_encoding = batch_data['question'].cuda()
            answers = batch_data['answers'].cuda()
            image = batch_data['image'].cuda()

            with torch.no_grad():
                predicted_answer = self._model(image, question_encoding)
                ground_truth_answer = torch.mean(answers, dim=1)
                loss += F.cross_entropy(predicted_answer, ground_truth_answer)
                pred_idx = torch.argmax(predicted_answer, dim=-1)
                gt_idx = torch.argmax(ground_truth_answer, dim=-1)

            correct += torch.sum(pred_idx == gt_idx)
            total += len(predicted_answer)
        loss = loss / total
        accuracy = correct / total
        # -----------------

        if self._log_validation:
            # ----------------- 2.9 TODO
            # you probably want to plot something here
            print("Epoch: {},  Val loss {}".format(epoch, loss.item()))

            # wandb
            if USE_WANDB:
                step_val += 1
                wandb.log({
                    "val_loss": loss,
                    "val_acc": accuracy,
                    "val_step": step_val,
                })

                for i in range(10):
                    image, question, pred_answer, gt_answer = self._visualize(batch_data, idxs[i], pred_idx[i], gt_idx[i])
                    self.text_table.add_data(epoch, step_val, question, [wandb.Image(image)], pred_answer, gt_answer, idxs[i])

            # tensorboard
            self.writer.add_scalar('Loss/Val', loss.item(), epoch)
            self.writer.add_scalar('Accuracy/Val', accuracy, epoch)

            images, questions, pred_answer, gt_answer = self._collect_data_for_visualization(batch_data, pred_idx, gt_idx)

            image_grid = torchvision.utils.make_grid(images)
            self.writer.add_image('Image', image_grid, epoch)
            self.writer.add_text('Question', questions, epoch)
            self.writer.add_text('Answer/Prediction', pred_answer, epoch)
            self.writer.add_text('Answer/GT', gt_answer, epoch)
            ############
        return accuracy

    def train(self):
        global step_train
        for epoch in tqdm(range(self._num_epochs)):
            num_batches = len(self._train_dataset_loader)
            for batch_id, batch_data in enumerate(self._train_dataset_loader):

                self._model.train()  # Set the model to train mode
                current_step = epoch * num_batches + batch_id

                # ----------------- 2.6 TODO
                # Run the model and get the ground truth answers that you'll pass to your optimizer
                # This logic should be generic; not specific to either the Simple Baseline or CoAttention.
                question_encoding = batch_data['question'].cuda()
                answers = batch_data['answers'].cuda()
                image = batch_data['image'].cuda()
                predicted_answer = self._model(image, question_encoding)  # TODO
                ground_truth_answer = answers  # TODO

                # -----------------
                # Optimize the model according to the predictions
                loss = self._optimize(predicted_answer, ground_truth_answer)

                if current_step % self._log_freq == 0:
                    print("Epoch: {}, Batch {}/{} has loss {}".format(epoch, batch_id, num_batches, loss.item()))
                    # ----------------- 2.9 TODO
                    # you probably want to plot something here
                    # -----------------
                    step_train += 1
                    if USE_WANDB:
                        wandb.log({
                            "train_loss": loss.item(),
                            "train_step": step_train
                        })

                    self.writer.add_scalar('Loss/Train', loss.item(), current_step // self._log_freq)
            if epoch % self._test_freq == 0:
                self._model.eval()
                val_accuracy = self.validate(epoch)
                print("Epoch: {} has val accuracy {}".format(epoch, val_accuracy))
                # ----------------- 2.9 TODO
                # you probably want to plot something here
                # -----------------
                # validation accuracy and loss are logged inside validate() method

            torch.save({
                'model_state_dict': self._model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'epoch': epoch,
            }, 'checkpoints/{}/epoch_{}.pth'.format(self.exp_name, epoch))
        wandb.log({
            "val_table": self.text_table,
        })
