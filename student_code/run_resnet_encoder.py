"""
Usage: python student_code/run_resnet_encoder.py --train_image_dir data/train2014 \
--train_question_path data/MultipleChoice_mscoco_train2014_questions.json \
--train_annotation_path data/mscoco_train2014_annotations.json \
--test_image_dir data/val2014 \
--test_question_path data/MultipleChoice_mscoco_val2014_questions.json \
--test_annotation_path data/mscoco_val2014_annotations.json \
"""

import argparse
import sys

sys.path.append("..")
from coattention_experiment_runner import CoattentionNetExperimentRunner

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Load VQA.')
    parser.add_argument('--train_image_dir', type=str, default='')
    parser.add_argument('--train_question_path', type=str, default='')
    parser.add_argument('--train_annotation_path', type=str)
    parser.add_argument('--test_image_dir', type=str)
    parser.add_argument('--test_question_path', type=str)
    parser.add_argument('--test_annotation_path', type=str)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--num_data_loader_workers', type=int, default=8)
    parser.add_argument('--cache_location', type=str, default='')
    parser.add_argument('--lr', type=float, default=4e-4)
    args = parser.parse_args()

    experiment_runner_class = CoattentionNetExperimentRunner

    experiment_runner = experiment_runner_class(train_image_dir=args.train_image_dir,
                                                train_question_path=args.train_question_path,
                                                train_annotation_path=args.train_annotation_path,
                                                test_image_dir=args.test_image_dir,
                                                test_question_path=args.test_question_path,
                                                test_annotation_path=args.test_annotation_path,
                                                batch_size=args.batch_size,
                                                num_epochs=args.num_epochs,
                                                num_data_loader_workers=args.num_data_loader_workers,
                                                cache_location=args.cache_location,
                                                lr=args.lr,
                                                log_validation=False)

    print(len(experiment_runner._train_dataset_loader))
    print(len(experiment_runner._val_dataset_loader))

    for batch_id, batch_data in enumerate(experiment_runner._train_dataset_loader):
        print('Loading training batches {}'.format(batch_id))
    for batch_id, batch_data in enumerate(experiment_runner._val_dataset_loader):
        print('Loading validation batches {}'.format(batch_id))
