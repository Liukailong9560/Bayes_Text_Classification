import jieba
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import pandas as pd

train_dic_path = 'text classification/train/'
test_dic_path = 'text classification/test/'
stop_words_path = 'text classification/stop/stopword.txt'

def cut_data(data, label):
    text = open(data, 'r', encoding='gb18030').read()
    text_cut = jieba.cut(text)
    text_with_space = ''
    for word in text_cut:
        text_with_space += word + ' '
    return text_with_space, label

def load_data(Path):
    data_list = []
    label_list = []
    for cls in os.listdir(Path):
        cls_path = os.path.join(Path, cls)
        for file in os.listdir(cls_path):
            file_path = os.path.join(cls_path, file)
            words, label = cut_data(file_path, cls)
            data_list.append(words)
            label_list.append(label)
    return data_list, label_list


if __name__ == '__main__':
    train_data_list, train_label_list = load_data(train_dic_path)
    test_data_list, test_label_list = load_data(test_dic_path)
    print(len(train_data_list), len(train_label_list), len(test_data_list), len(test_label_list))

    stop_words = open(stop_words_path, 'r', encoding='utf-8').read()
    stop_words = stop_words.encode('utf-8').decode('utf-8-sig')  # 列表头部\ufeff处理
    stop_words = stop_words.split('\n')  # 转化为列表

    tf = TfidfVectorizer(stop_words=stop_words, max_df=0.5)
    train_features = tf.fit_transform(train_data_list)  # 得到每个Word的TF-IDF值
    test_features = tf.transform(test_data_list)
    df = pd.DataFrame(train_features.toarray(), columns=tf.get_feature_names())
    print(df.head())

    # BayesClassifier
    bayesClassifier = MultinomialNB(alpha=0.001)  # Lidstone平滑
    # from sklearn.calibration import CalibratedClassifierCV  # 概率校准
    # bayesClassifier = CalibratedClassifierCV(MultinomialNB(), cv=2, method='sigmoid')
    bayesClassifier.fit(train_features, train_label_list)
    pred = bayesClassifier.predict(test_features)

    # 计算正确率
    from sklearn import metrics
    print('Accuracy:', metrics.accuracy_score(test_label_list, pred))