from torch.utils.data import DataLoader
import torch
import os
import wandb
from tqdm import tqdm
import random
USE_WANDB = False
step_train = 0
step_val = 0
from PIL import Image

class ExperimentRunnerBase(object):
    """
    This base class contains the simple train and validation loops for your VQA experiments.
    Anything specific to a particular experiment (Simple or Coattention) should go in the corresponding subclass.
    """

    def __init__(self, train_dataset, val_dataset, model, batch_size, num_epochs, num_data_loader_workers=4, log_validation=True):
        self._model = model
        self._num_epochs = num_epochs
        self._log_freq = 200 #10  # Steps
        self._test_freq = 250  # Steps
        self.val_dataset = val_dataset
        self._train_dataset_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_data_loader_workers)

        # If you want to, you can shuffle the validation dataset and only use a subset of it to speed up debugging
        self._val_dataset_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_data_loader_workers)

        # Use the GPU if it's available.
        self._cuda = torch.cuda.is_available()

        if self._cuda:
            self._model = self._model.cuda()

        self._log_validation = log_validation

        if USE_WANDB:
            wandb.init(
                project="vlr_hw4",
                name=f"simple_baseline"
            )
            self.text_table = wandb.Table(columns=["epoch", "val_step", "question",  "image", "pred_answer", "gt_answer", "sample_idx"])

    def _visualize(self, idx, pred_idx, gt_idx):

        import pdb; pdb.set_trace()
        if torch.is_tensor(idx):
            idx = idx.tolist()

        fname = self.val_dataset.index_list[idx]
        qid = int(fname)
        image_id = self.val_dataset._vqa.qa[qid]['image_id']
        image_id = '{:0>12}'.format(image_id)

        # image
        image_path = os.path.join(
                self.val_dataset._image_dir, self.val_dataset._image_filename_pattern.format(image_id))
        image = Image.open(image_path).convert('RGB')

        #question
        question = self.val_dataset._vqa.qqa[qid]['question']

        #gt_answer
        gt_answer = self.val_dataset.answer_word_list[gt_idx]

        #pred_answer
        pred_answer = self.val_dataset.answer_word_list[pred_idx]
        return image, question, pred_answer, gt_answer
        # self.val_dataset.visualize = True
        # sample = self.val_dataset[idx]
        # self.val_dataset.visualize = False
        
        # image = sample['image']
        # question = sample['question']
        # answers = sample['answers']
        # majority_vote_answer =  Counter(answers).most_common[1][0][0]
        # return image, question, answer

    def _optimize(self, predicted_answers, true_answers):
        """
        This gets implemented in the subclasses. Don't implement this here.
        """
        raise NotImplementedError()

    def validate(self, epoch):
        # ----------------- 2.8 TODO
        # Should return your validation accuracy
        global step_val
        correct, total = 0.0, 0.0
        # TODO fix accuracy calculation
        idxs, pred_idx, gt_idx = None, None, None
        print(f"running validation for epoch: {epoch}")
        for batch_id, batch_data in enumerate(self._val_dataset_loader):
            step_val += 1
            idxs = batch_data['idx']
            question_encoding = batch_data['question'].cuda()
            answers = batch_data['answers'].cuda()
            image = batch_data['image'].cuda()

            predicted_answer = self._model(image, question_encoding)
            ground_truth_answer = answers
            loss = self._optimize(predicted_answer, ground_truth_answer)

            ground_truth_answer = torch.sum(ground_truth_answer, dim=1).squeeze()
            pred_idx = torch.argmax(predicted_answer, dim=-1)
            gt_idx = torch.argmax(ground_truth_answer, dim = -1)
            correct += torch.sum(pred_idx == gt_idx)
            total += len(predicted_answer)
            accuracy = correct / total
        # -----------------

        if self._log_validation:
            # ----------------- 2.9 TODO
            # you probably want to plot something here
            print("Epoch: {},  Val loss {}".format(epoch, loss))
            image, question, pred_answer, gt_answer = self._visualize(idxs[0], pred_idx[0], gt_idx[0])
            # image, question, pred_answer, gt_answer = self._visualize(idxs)
            # -----------------
            if USE_WANDB:
                wandb.log({ 
                    "val_loss": loss, 
                    "val_acc" : accuracy,
                    "val_step": step_val,
                    })
                self.text_table.add_data(epoch, step_val, question, [wandb.Image(image)], pred_answer, gt_answer, idxs[0])
        return accuracy

    def train(self):
        global step_train
        for epoch in tqdm(range(self._num_epochs)):
            num_batches = len(self._train_dataset_loader)
            for batch_id, batch_data in enumerate(self._train_dataset_loader):
                
                step_train += 1
                self._model.train()  # Set the model to train mode
                current_step = epoch * num_batches + batch_id

                # ----------------- 2.6 TODO
                # Run the model and get the ground truth answers that you'll pass to your optimizer
                # This logic should be generic; not specific to either the Simple Baseline or CoAttention.
                question_encoding = batch_data['question'].cuda()
                answers = batch_data['answers'].cuda()
                image = batch_data['image'].cuda()
                predicted_answer = self._model(image, question_encoding) # TODO
                ground_truth_answer = answers # TODO

                # -----------------
                # Optimize the model according to the predictions
                loss = self._optimize(predicted_answer, ground_truth_answer)

                if current_step % self._log_freq == 0:
                    print("Epoch: {}, Batch {}/{} has loss {}".format(epoch, batch_id, num_batches, loss.item()))
                    # ----------------- 2.9 TODO
                    # you probably want to plot something here
                    # -----------------
                    if USE_WANDB:    
                        wandb.log({ 
                            "train_loss": loss, 
                            "train_step": step_train
                            })

                if current_step % self._test_freq == 0:
                    self._model.eval()
                    val_accuracy = self.validate(epoch)
                    print("Epoch: {} has val accuracy {}".format(epoch, val_accuracy))
                    # ----------------- 2.9 TODO
                    # you probably want to plot something here
                    # -----------------
                    # validation accuracy and loss are logged inside validate() method
        wandb.log({
            "val_table" : self.text_table,
            })