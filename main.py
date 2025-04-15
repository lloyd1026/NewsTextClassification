import os
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.linear_model import SGDClassifier

# import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset


# 读取类别标签
def load_classes(class_file):
    with open(class_file, 'r', encoding='utf-8') as f:
        classes = [line.strip() for line in f]
    label_to_id = {label: idx for idx, label in enumerate(classes)}
    id_to_label = {idx: label for label, idx in label_to_id.items()}
    return classes, label_to_id, id_to_label


# 读取数据文件（标题\t类别）
def load_data(file_path):
    texts = []
    labels = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if '\t' not in line:
                continue
            title, label = line.strip().split('\t')
            texts.append(' '.join(jieba.cut(title)))  # 分词
            labels.append(label)
    return texts, labels


def sgd_classifier(x_train_tfidf, y_train, x_val_tfidf, y_val, x_test_tfidf, y_test, classes):
    # 传统机器学习模型——凸优化问题，有最优解
    print("📌 使用 SVM 模型进行训练...")
    # clf = SVC(kernel='linear', C=1.0)
    # clf.fit(x_train_tfidf, y_train)
    clf = SGDClassifier(loss='hinge', max_iter=1000, tol=1e-3, verbose=1)  # SVM 损失函数
    clf.fit(x_train_tfidf, y_train)

    print("✅ 在验证集上评估：")
    y_val_pred = clf.predict(x_val_tfidf)
    print(classification_report(y_val, y_val_pred, target_names=classes))
    print("Accuracy:", accuracy_score(y_val, y_val_pred))

    print("\n✅ 在测试集上评估：")
    y_test_pred = clf.predict(x_test_tfidf)
    print(classification_report(y_test, y_test_pred, target_names=classes))
    print("Accuracy:", accuracy_score(y_test, y_test_pred))


def encode_data(texts, labels, tokenizer, max_length=128):
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length)
    encodings['labels'] = labels
    return encodings


def bert_classifier(x_train, y_train, x_val, y_val, x_test, y_test, classes):
    print("📌 使用 BERT 模型进行训练...")
    # 加载 BERT 的 Tokenizer 和 模型
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    model = BertForSequenceClassification.from_pretrained('bert-base-chinese', num_labels=len(classes))

    # 将数据转换为 BERT 输入格式
    train_encodings = encode_data(x_train, y_train, tokenizer)
    val_encodings = encode_data(x_val, y_val, tokenizer)
    test_encodings = encode_data(x_test, y_test, tokenizer)

    # 转换为 HuggingFace Dataset 格式
    train_dataset = Dataset.from_dict(train_encodings)
    val_dataset = Dataset.from_dict(val_encodings)
    test_dataset = Dataset.from_dict(test_encodings)

    # 训练配置
    training_args = TrainingArguments(
        output_dir='./results',          
        num_train_epochs=3,             
        per_device_train_batch_size=8,   
        per_device_eval_batch_size=8,    
        warmup_steps=500,               
        weight_decay=0.01,              
        logging_dir='./logs',            
        logging_steps=1000,
    )

    trainer = Trainer(
        model=model, 
        args=training_args,
        train_dataset=train_dataset, 
        eval_dataset=val_dataset
    )

    print("📌 开始训练...")
    trainer.train()

    print("📌 在验证集上评估：")
    val_results = trainer.evaluate(val_dataset)
    print(f"Validation loss: {val_results['eval_loss']}")
    print(f"Validation accuracy: {val_results['eval_accuracy']}")

    # 在测试集上评估
    print("\n📌 在测试集上评估：")
    test_results = trainer.evaluate(test_dataset)
    print(f"Test loss: {test_results['eval_loss']}")
    print(f"Test accuracy: {test_results['eval_accuracy']}")

    # 获取预测结果
    print("\n📌 在测试集上进行预测：")
    predictions = trainer.predict(test_dataset)
    preds = predictions.predictions.argmax(axis=-1)

    # 打印分类报告
    print(classification_report(y_test, preds, target_names=classes))
    print("Accuracy:", accuracy_score(y_test, preds))


def main():
    base_dir = os.path.dirname(__file__)
    class_file = os.path.join(base_dir, 'class.txt')
    train_file = os.path.join(base_dir, 'train.txt')
    val_file = os.path.join(base_dir, 'val.txt')
    test_file = os.path.join(base_dir, 'test.txt')

    print("📌 加载类别映射...")
    classes, label_to_id, id_to_label = load_classes(class_file)

    print("📌 加载训练数据...")
    x_train, y_train_labels = load_data(train_file)
    print("📌 加载验证数据...")
    x_val, y_val_labels = load_data(val_file)
    print("📌 加载测试数据...")
    x_test, y_test_labels = load_data(test_file)

    # type(id) 'str' -> 'int'
    try:
        y_train = [int(label) for label in y_train_labels]
        y_val = [int(label) for label in y_val_labels]
        y_test = [int(label) for label in y_test_labels]
    except KeyError as e:
        print(e)
        exit(1)

    print("📌 构建 TF-IDF 特征...")
    vectorizer = TfidfVectorizer(max_features=10000)
    x_train_tfidf = vectorizer.fit_transform(x_train)
    x_val_tfidf = vectorizer.transform(x_val)
    x_test_tfidf = vectorizer.transform(x_test)

    # 调用 SVM 分类器
    # sgd_classifier(x_train_tfidf, y_train, x_val_tfidf, y_val, x_test_tfidf, y_test, classes)

    # 调用 BERT 分类器
    bert_classifier(x_train, y_train, x_val, y_val, x_test, y_test, classes)


if __name__ == '__main__':
    main()