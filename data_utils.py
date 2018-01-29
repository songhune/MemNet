from __future__ import absolute_import

import os
import re
import numpy as np


def load_task(data_dir, task_id, only_supporting=False):
    '''Load the nth task. There are 20 tasks in total.

    Returns a tuple containing the training and testing data for the task.
    '''
    assert task_id > 0 and task_id < 21

    files = os.listdir(data_dir)
    files = [os.path.join(data_dir, f) for f in files]
    s = 'qa{}_'.format(task_id)
    train_file = [f for f in files if s in f and 'train' in f][0]
    test_file = [f for f in files if s in f and 'test' in f][0]
    train_data = get_stories(train_file, only_supporting)
    test_data = get_stories(test_file, only_supporting)
    return train_data, test_data


def tokenize(sent):
    '''Return the tokens of a sentence including punctuation.
    >>> tokenize('Bob dropped the apple. Where is the apple?')
    ['Bob', 'dropped', 'the', 'apple', '.', 'Where', 'is', 'the', 'apple', '?']
    '''
    return [x.strip() for x in re.split('(\\W+)?', sent) if x.strip()]


def parse_stories(lines, only_supporting=False):
    '''Parse stories provided in the bAbI tasks format
    If only_supporting is true, only the sentences that support the answer are kept.
    '''
    data = []
    story = []
    for line in lines:
        line = str.lower(line)  # 다 소문자로 바꾼다
        nid, line = line.split(' ', 1)
        nid = int(nid)
        if nid == 1:
            story = []
            # bAbI 특성상 연관된 문장이 끝나면 다시 1번부터 번호를 매기기 때문에 이런 식으로 구분해준것
        if '\t' in line:  # question이라면
            q, a, supporting = line.split('\t')  # 나누고
            q = tokenize(q)  # 쪼갠다
            # a = tokenize(a)
            # answer is one vocab word even if it's actually multiple words??? 이게 뭐지?
            a = [a]
            substory = None

            # remove question marks
            if q[-1] == "?":
                q = q[:-1]

            if only_supporting:
                # Only select the related substory
                supporting = map(int, supporting.split())
                substory = [story[i - 1] for i in supporting]
            else:
                # Provide all the substories
                substory = [x for x in story if x]

            data.append((substory, q, a))
            story.append('')
        else:  # regular sentence
            # remove periods
            sent = tokenize(line)
            if sent[-1] == ".":
                sent = sent[:-1]
            story.append(sent)
    return data


def get_stories(f, only_supporting=False):
    '''Given a file name, read the file, retrieve the stories, and then convert the sentences into a single story.
    If max_length is supplied, any stories longer than max_length tokens will be discarded.
    '''
    with open(f) as f:
        return parse_stories(f.readlines(), only_supporting=only_supporting)


def vectorize_data(data, word_idx, sentence_size, memory_size, max_story_size):
    """
    Vectorize stories and queries.

    If a sentence length < sentence_size, the sentence will be padded with 0's.
         문장의 길이        문장의 크기, 즉 변수
    If a story length < memory_size, the story will be padded with empty memories.
    Empty memories are 1-D arrays of length sentence_size filled with 0's.

    songhune edited: if a story length > memory size, max size of each story will be returned.
    The answer array is returned as a one-hot encoding.
    """
    S = []
    Q = []
    A = []
    for story, query, answer in data:
        ss = []
        for i, sentence in enumerate(story, 1):
            # 1부터 시작한다. #ls는 sentence size(7개)에서 현재 문장이 들어가 있는 만큼의 공간을 빼줌
            ls = max(0, sentence_size - len(sentence))
            ss.append([word_idx[w] for w in sentence] + [0] * ls)  # 뭘 붙이냐 하면 word index를 붙인다.

        # take only the most recent sentences that fit in memory
        '''일단 전체 메모리를 뱉어내게 한다. '''
        ss = ss[::-1][:max_story_size][::-1]
        # 역수로 출력 / 전체 크기만큼 / 역수로 출력
        '''
        if story_size<memory_size
            ss =[::-1][:memory_size][::-1]

        elif story_size>memory_size
            if memory full
            ss = 기존
        이럴 필요가 있는가?
        '''
        # Make the last word of each sentence the time 'word' which corresponds to vector of lookup table
        for i in range(len(ss)):
            # 30에서 메모리 사이즈 10개 빼고 i만큼 빼고, ss의 길이를 붙인다. 룩업 테이블과 연관됨
            ss[i][-1] = len(word_idx) - max_story_size - i + len(ss)

        # pad  '0' to memory_size
        # songhune edited: 메모리 사이즈에서 story size를 빼는 것이 아니라 전체 크기에서 메모리 사이즈를 반환한다.
        lm = max(0, max_story_size - len(ss))
        for _ in range(lm):
            ss.append([0] * sentence_size)

        # manipulate query
        lq = max(0, sentence_size - len(query))
        q = [word_idx[w] for w in query] + [0] * lq

        y = np.zeros(len(word_idx) + 1)  # 0 is reserved for nil word
        for a in answer:
            y[word_idx[a]] = 1

        S.append(ss)
        Q.append(q)
        A.append(y)
    print(np.array(S).shape)
    print(np.array(Q).shape)
    print(np.array(A).shape)
    return np.array(S), np.array(Q), np.array(A)
