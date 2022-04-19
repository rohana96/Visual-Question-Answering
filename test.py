import string
from collections import Counter
from student_code.vqa_dataset import VqaDataset 

def _create_id_map(word_list, max_list_length):
    """
    Find the most common str in a list, then create a map from str to id (its rank in the frequency)
    Args:
        word_list: a list of str, where the most frequent elements are picked out
        max_list_length: the number of strs picked
    Return:
        A map (dict) from str to id (rank)
    """

    ############ 1.5 TODO
    word_ranking = {}
    counter = Counter(word_list)
    freq_word = counter.most_common(max_list_length)
    for i, word in enumerate(freq_word):
        word_ranking[word[0]] = i
    return word_ranking
    ############
def create_word_list(sentences):
    """
    Turn a list of sentences into a list of processed words (no punctuation, lowercase, etc)
    Args:
        sentences: a list of str, sentences to be splitted into words
    Return:
        A list of str, words from the split, order remained.
    """

    ############ 1.4 TODO
    word_list = []
    for sentence in sentences:
        sentence_no_punc = sentence.translate(str.maketrans('', '', string.punctuation))
        sentence_no_punc = sentence_no_punc.lower()
        words = sentence_no_punc.split()
        word_list.extend(words)

    ############
    return word_list

def test_create_word():
    sentences = [
        "hello the! ,cute   ., we. . .",
        "bweer! 9 efwqjl4,3s"
    ]
    word_list = create_word_list(sentences)
    print(word_list)
    word_ranking = _create_id_map(word_list, 5)
    print(word_ranking)


def test_VQADataset():

    dataset = VqaDataset(
        image_dir='data/train2014', 
        question_json_file_path ='data/OpenEnded_mscoco_train2014_questions.json',
        annotation_json_file_path = 'data/mscoco_train2014_annotations.json', 
        image_filename_pattern="COCO_train2014_{}.jpg"
        )

    sample = dataset[40500]
    question = sample['question']
    answer = sample['answers']
    print(question.shape)
    print(question[question == 1].shape)
    print(answer[0][answer[0] == 1].shape)
    print("end")

if __name__ == "__main__":
    test_VQADataset()