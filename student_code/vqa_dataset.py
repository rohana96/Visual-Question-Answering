import os
import torch

from PIL import Image
from torch.utils.data import Dataset
from VQA.external.vqa.vqa import VQA
import string
from collections import Counter
from tqdm import tqdm
from torchvision.transforms import transforms
    


class VqaDataset(Dataset):
    """
    Load the VQA dataset using the VQA python API. We provide the necessary subset in the External folder, but you may
    want to reference the full repo (https://github.com/GT-Vision-Lab/VQA) for usage examples.
    """

    def __init__(self, image_dir, question_json_file_path, annotation_json_file_path, image_filename_pattern,
                 transform=None, question_word_to_id_map=None, answer_to_id_map=None, id_to_answer_map=None, question_word_list_length=5746, answer_list_length=5216,
                 pre_encoder=None, cache_location=None, answer_word_list=None):
        """
        Args:
            image_dir (string): Path to the directory with COCO images
            question_json_file_path (string): Path to the json file containing the question data
            annotation_json_file_path (string): Path to the json file containing the annotations mapping images, questions, and
                answers together
            image_filename_pattern (string): The pattern the filenames of images in this dataset use (eg "COCO_train2014_{}.jpg")
        """
        self._vqa = VQA(annotation_file=annotation_json_file_path, question_file=question_json_file_path)
        self._image_dir = image_dir
        self._image_filename_pattern = image_filename_pattern
        self._transform = transform
        self._max_question_length = 26
        self.visualize = False  # to return PIL images and questions and answers in string format for visualization

        # Publicly accessible dataset parameters
        self.question_word_list_length = question_word_list_length + 1  # +1 for 'unknown' category (out of vocab words)
        self.unknown_question_word_index = question_word_list_length
        self.answer_list_length = answer_list_length + 1
        self.unknown_answer_index = answer_list_length
        self._pre_encoder = pre_encoder
        self._cache_location = cache_location
        if self._cache_location is not None:
            try:
                os.makedirs(self._cache_location)
            except OSError:
                pass

        self.num_images = len(os.listdir(self._image_dir))
        self.num_questions = len(self._vqa.qqa.keys())
        self.index_list = [str(k) for k,_ in self._vqa.qqa.items()]

        # Create the question map if necessary
        if question_word_to_id_map is None:
            # ----------------- 1.6 TODO
            sentences = []
            questions_info_dict = self._vqa.questions
            questions_dict_list = questions_info_dict['questions']
            for question_dict in tqdm(questions_dict_list):
                sentences.append(question_dict['question'])
                
            question_word_list = self._create_word_list(sentences)      
            question_word_to_id_map = self._create_id_map(question_word_list, self.question_word_list_length - 1)  # last index reserved for 'unknown'
            question_word_to_id_map['out_of_vocab'] = self.unknown_question_word_index
            self.question_word_to_id_map = question_word_to_id_map
            print(f"Finished creating question map: {len(question_word_to_id_map.keys())} total question words")
            # -----------------
        else:
            self.question_word_to_id_map = question_word_to_id_map

        # Create the answer map if necessary

        if answer_to_id_map is None:
            # ----------------- 1.7 TODO
            answer_word_list = []
            qids = self._vqa.getQuesIds()
            for qid in tqdm(qids):
                answer_dict_list = self._vqa.qa[qid]['answers']
                for answer_dict in answer_dict_list:
                    answer = answer_dict['answer']
                    answer_word_list.append(answer)

            answer_to_id_map, id_to_answer_map = self._create_id_map(answer_word_list, self.answer_list_length - 1, return_reversed_map=True)  # last index reserved for 'unknown'
            
            answer_to_id_map['out_of_vocab'] = self.unknown_answer_index
            id_to_answer_map[self.unknown_answer_index] = 'out_of_vocab'
            
            self.answer_to_id_map = answer_to_id_map
            self.id_to_answer_map = id_to_answer_map
            self.answer_word_list = answer_word_list

            print(f"Finished creating answers map: {len(answer_to_id_map.keys())} total answers")
            # -----------------
        else:
            self.id_to_answer_map = id_to_answer_map
            self.answer_to_id_map = answer_to_id_map
            self.answer_word_list = answer_word_list


    def _create_word_list(self, sentences):
        """
        Turn a list of sentences into a list of processed words (no punctuation, lowercase, etc)
        Args:
            sentences: a list of str, sentences to be splitted into words
        Return:
            A list of str, words from the split, order remained.
        """
        # ----------------- 1.4 TODO
        word_list = []
        for sentence in sentences:
            try:
                sentence_no_punc = sentence.translate(str.maketrans('', '', string.punctuation))
            except:
                import pdb; pdb.set_trace()
            sentence_no_punc = sentence_no_punc.lower()
            words = sentence_no_punc.split()
            word_list.extend(words)

        # -----------------
        return word_list


    def _create_id_map(self, word_list, max_list_length, return_reversed_map=False):
        """
        Find the most common str in a list, then create a map from str to id (its rank in the frequency)
        Args:
            word_list: a list of str, where the most frequent elements are picked out
            max_list_length: the number of strs picked
        Return:
            A map (dict) from str to id (rank)
        """

        # ----------------- 1.5 TODO
        word_ranking = {}
        counter = Counter(word_list)
        freq_word = counter.most_common(max_list_length) 
        for i, word in enumerate(freq_word):
            word_ranking[word[0]] = i
        
        if return_reversed_map:
            ranking_to_word = {}
            for word, rank in word_ranking.items():
                ranking_to_word[rank] = word
            return word_ranking, ranking_to_word

        return word_ranking
        # -----------------
    
    def _one_hot_encode(self, word_list, vocab_dict, max_list_length):
        """
        Returns one hot encoding of the input wordlist
        Args:
            word_list: a list of str
            type: indicates if it is a question or an answer
        Return:
            one hot vector
        """
        one_hot_vector = torch.zeros(max_list_length)
        for word in word_list:
            if word in vocab_dict:
                one_hot_vector[vocab_dict[word]] = 1
            else:
                one_hot_vector[vocab_dict['out_of_vocab']] = 1
        return one_hot_vector.reshape(1, -1)

    def __len__(self):
        # ----------------- 1.8 TODO
        return self.num_questions
        # -----------------
        
    def __getitem__(self, idx):
        """
        Load an item of the dataset
        Args:
            idx: index of the data item
        Return:
            A dict containing multiple torch tensors for image, question and answers.
        """
        # ----------------- 1.9 TODO
        # figure out the idx-th item of dataset from the VQA API
        if torch.is_tensor(idx):
            idx = idx.tolist()
        fname = self.index_list[idx]
        qid = int(fname)
        image_id = self._vqa.qa[qid]['image_id']
        image_id = '{:0>12}'.format(image_id)
        # -----------------

        if self._cache_location is not None and self._pre_encoder is not None:
            # the caching and loading logic here
            feat_path = os.path.join(self._cache_location, f'{image_id}.pt')
            try:
                image = torch.load(feat_path)
            except:
                image_path = os.path.join(
                    self._image_dir, self._image_filename_pattern.format(image_id))
                image = Image.open(image_path).convert('RGB')
                image = self._transform(image).unsqueeze(0)
                image = self._pre_encoder(image)[0]
                torch.save(image, feat_path)
        else:
            # ----------------- 1.9 TODO
            # load the image from disk, apply self._transform (if not None)
            image_path = os.path.join(
                    self._image_dir, self._image_filename_pattern.format(image_id))
            image_PIL = Image.open(image_path).convert('RGB')

            if self._transform:
                _transform = self._transform
            else:
                _transform = transforms.ToTensor()

            image = _transform(image_PIL)
            # -----------------

        # ----------------- 1.9 TODO
        # load and encode the question and answers, convert to torch tensors

        question = self._vqa.qqa[qid]['question']
        question_word_list = self._create_word_list([question])[:self._max_question_length]
        question_tensor = self._one_hot_encode(question_word_list, self.question_word_to_id_map, self.question_word_list_length)

        answer_one_hot_list = []
        answers = []
        answer_dict_list = self._vqa.qa[qid]['answers']
        for answer_dict in answer_dict_list:
            answer = answer_dict['answer']
            answers.append(answer)
            answer_one_hot = self._one_hot_encode([answer], self.answer_to_id_map, self.answer_list_length)
            answer_one_hot_list.append(answer_one_hot)
        answers_tensor = torch.cat(answer_one_hot_list, dim=0)
        # -----------------
        if not self.visualize:
            return {
                'idx': idx,
                'image': image,
                'question': question_tensor,
                'answers': answers_tensor
            }

        return 
        {
            'idx': idx, 
            'image': image_PIL,
            'question': question,
            'answers': answers
        }





