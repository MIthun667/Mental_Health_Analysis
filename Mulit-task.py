import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ast
from collections import Counter
import re
from transformers import BertTokenizer
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Layer, Dense, Dropout, LayerNormalization, MultiHeadAttention, Embedding
from tensorflow.keras.initializers import RandomNormal, Zeros
from tensorflow.keras.layers import Input, Conv1D,  MaxPooling1D, Bidirectional, GRU, Dense, Dropout, LayerNormalization, Embedding, Concatenate
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA



balanced_df = pd.read_csv("./Balanced.csv")
balanced_df.head()

balanced_df['opinions_str'] = balanced_df['opinions'].apply(lambda x: ','.join(x))


if isinstance(balanced_df['opinions'][0], str):
    balanced_df['opinions'] = balanced_df['opinions'].apply(ast.literal_eval)

all_opinions = [opinion for sublist in balanced_df['opinions'] for opinion in sublist]

opinion_counts = Counter(all_opinions)
most_common_opinions = opinion_counts.most_common(10)
print("Most frequent opinions:", most_common_opinions)

# 1. Total number of unique opinions
all_opinions = [opinion for sublist in balanced_df['opinions'] for opinion in sublist]
unique_opinions = set(all_opinions)
print(f"Total unique opinions: {len(unique_opinions)}")

sentiment_df = balanced_df[["statement", "status", "sentiment", "opinions", ]]

sentiment_df.isnull().sum()

# Drop rows with missing values
sentiment_df = sentiment_df.dropna()

def clean_opinions(opinions):
    cleaned_opinions = []
    for opinion in opinions:
        cleaned = re.sub(r'[^a-zA-Z\s]', '', opinion) 
        if cleaned and len(cleaned) > 1 and not re.match(r'^[a-zA-Z]*([a-zA-Z])\1{2,}[a-zA-Z]*$', cleaned):
            cleaned_opinions.append(cleaned.lower())  
    return cleaned_opinions

sentiment_df['cleaned_opinions'] = sentiment_df['opinions'].apply(clean_opinions)
sentiment_df['opinions_str'] = sentiment_df['cleaned_opinions'].apply(lambda x: ' '.join(x))

print(sentiment_df['status'].dtype)
print(sentiment_df['sentiment'].dtype)
print(sentiment_df['opinions'].dtype)

# Ensure the columns are of type string
sentiment_df['status'] = sentiment_df['status'].astype(str)
sentiment_df['sentiment'] = sentiment_df['sentiment'].astype(str)
sentiment_df['opinions'] = sentiment_df['opinions'].apply(lambda x: ' '.join(x) if isinstance(x, list) else str(x))


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize the input data
def tokenize_data(sentences, tokenizer, max_length=100):
    encoded_data = tokenizer.batch_encode_plus(
        sentences,
        add_special_tokens=True, 
        max_length=max_length,            
        padding='max_length',            
        truncation=True,                  
        return_attention_mask=True,      
        return_tensors='tf'              
    )
    return encoded_data['input_ids'], encoded_data['attention_mask']

sentiment_df['sentiment'] = sentiment_df['sentiment'].apply(lambda x: str(x))

# Apply tokenizer to 'statement' column
input_ids, attention_masks = tokenize_data(sentiment_df['statement'].tolist(), tokenizer)


sentiment_df['opinions_str'] = sentiment_df['opinions'].apply(lambda x: ' '.join(x) if isinstance(x, list) else str(x))

# Initialize label encoders
label_encoder_status = LabelEncoder()
label_encoder_sentiment = LabelEncoder()
label_encoder_opinions = LabelEncoder()

# Fit and transform the labels to integers
sentiment_df['status_encoded'] = label_encoder_status.fit_transform(sentiment_df['status'])
sentiment_df['sentiment_encoded'] = label_encoder_sentiment.fit_transform(sentiment_df['sentiment'])
sentiment_df['opinions_encoded'] = label_encoder_opinions.fit_transform(sentiment_df['opinions_str'])

# Get the unique integer classes for opinions
unique_opinions_count = len(label_encoder_opinions.classes_)
print("Unique opinions classes:", unique_opinions_count)

# Encode labels for status and sentiment
status_labels = to_categorical(sentiment_df['status_encoded'])
sentiment_labels = to_categorical(sentiment_df['sentiment_encoded'])

# For opinions, use integer encoding directly in the model
opinions_labels = sentiment_df['opinions_encoded']

# Display the updated DataFrame with encoded labels
print(sentiment_df[['status', 'status_encoded', 'sentiment', 'sentiment_encoded', 'opinions_str', 'opinions_encoded']])

# Printing the shapes to confirm the tensors
print("Input IDs shape:", input_ids.shape)
print("Attention Masks shape:", attention_masks.shape)

# If you want to inspect the first few rows
print("Input IDs:", input_ids[:5])
print("Attention Masks:", attention_masks[:5])


# Convert TensorFlow tensors to numpy arrays
input_ids_np = input_ids.numpy()
attention_masks_np = attention_masks.numpy()
status_labels_np = status_labels
sentiment_labels_np = sentiment_labels
opinions_labels_np = opinions_labels

# Split data into training, validation, and test sets
X_train, X_temp, attention_train, attention_temp, y_status_train, y_status_temp, y_sentiment_train, y_sentiment_temp, y_opinions_train, y_opinions_temp = train_test_split(
    input_ids_np, attention_masks_np, status_labels_np, sentiment_labels_np, opinions_labels_np, 
    test_size=0.3, random_state=42
)

X_val, X_test, attention_val, attention_test, y_status_val, y_status_test, y_sentiment_val, y_sentiment_test, y_opinions_val, y_opinions_test = train_test_split(
    X_temp, attention_temp, y_status_temp, y_sentiment_temp, y_opinions_temp, 
    test_size=0.5, random_state=42
)

# Print shapes to verify correct splitting
print(f"Training set: {X_train.shape[0]} samples")
print(f"Validation set: {X_val.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")


# Token and Positional Embedding
class TokenAndPositionEmbedding(Layer):
    def __init__(self, maxlen, vocab_size, embed_dim, use_sinusoidal=False, dropout_rate=0.1, **kwargs):
        """
        Args:
            maxlen: Maximum length of the input sequences.
            vocab_size: Size of the vocabulary.
            embed_dim: Dimension of the embeddings.
            use_sinusoidal: Whether to use sinusoidal positional embeddings.
            dropout_rate: Dropout rate for regularization.
        """
        super(TokenAndPositionEmbedding, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.maxlen = maxlen
        self.use_sinusoidal = use_sinusoidal
        
        self.token_emb = Embedding(input_dim=vocab_size, output_dim=embed_dim, embeddings_initializer=RandomNormal())
        self.pos_emb = Embedding(input_dim=maxlen, output_dim=embed_dim, embeddings_initializer=RandomNormal())
        self.dropout = Dropout(dropout_rate)
        self.layer_norm = LayerNormalization(epsilon=1e-6)
        
        if use_sinusoidal:
            self.pos_emb = self._create_sinusoidal_embeddings(maxlen, embed_dim)

    def _create_sinusoidal_embeddings(self, maxlen, embed_dim):
        """Create sinusoidal positional embeddings."""
        position = tf.range(0, maxlen, dtype=tf.float32)
        i = tf.range(0, embed_dim, 2, dtype=tf.float32) / tf.cast(embed_dim, tf.float32)
        angle_rates = 1 / tf.pow(10000.0, i)
        angle_rads = tf.expand_dims(position, 1) * tf.expand_dims(angle_rates, 0)
        
        sines = tf.math.sin(angle_rads)
        cosines = tf.math.cos(angle_rads)
        
        pos_encoding = tf.concat([sines, cosines], axis=-1)
        pos_encoding = tf.expand_dims(pos_encoding, 0)
        
        return tf.constant(pos_encoding, dtype=tf.float32)

    def call(self, x):
        """
        Forward pass for the embedding layer.

        Args:
            x: Input tensor of shape (batch_size, seq_len).
        
        Returns:
            Output tensor with added token and positional embeddings.
        """
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)

        if self.use_sinusoidal:
            pos_emb = self.pos_emb[:, :maxlen, :]
        else:
            pos_emb = self.pos_emb(positions)

        x = self.token_emb(x)
        x += pos_emb

        x = self.layer_norm(x)
        x = self.dropout(x)
        
        return x

    def get_config(self):
        config = super(TokenAndPositionEmbedding, self).get_config()
        config.update({
            "maxlen": self.maxlen,
            "vocab_size": self.token_emb.input_dim,
            "embed_dim": self.embed_dim,
            "use_sinusoidal": self.use_sinusoidal,
            "dropout_rate": self.dropout.rate
        })
        return config

# Opinion Embeddings
class OpinionsEmbedding(Layer):
    def __init__(self, attention_dim, num_heads=2, use_scale=False, **kwargs):
        """
        Args:
            attention_dim: Dimensionality of the attention mechanism.
            num_heads: Number of attention heads.
            use_scale: Whether to scale the attention scores.
        """
        super(OpinionsEmbedding, self).__init__(**kwargs)
        self.attention_dim = attention_dim
        self.num_heads = num_heads
        self.use_scale = use_scale
        self.query_dense_layers = []
        self.key_dense_layers = []
        self.value_dense_layers = []

    def build(self, input_shape):
        for _ in range(self.num_heads):
            self.query_dense_layers.append(Dense(self.attention_dim, kernel_initializer=RandomNormal(), bias_initializer=Zeros(), name=f"query_{_}"))
            self.key_dense_layers.append(Dense(self.attention_dim, kernel_initializer=RandomNormal(), bias_initializer=Zeros(), name=f"key_{_}"))
            self.value_dense_layers.append(Dense(self.attention_dim, kernel_initializer=RandomNormal(), bias_initializer=Zeros(), name=f"value_{_}"))

        self.Wa = self.add_weight(shape=(self.attention_dim, self.attention_dim),
                                  initializer=RandomNormal(),
                                  trainable=True,
                                  name="Wa")
        self.Ua = self.add_weight(shape=(self.attention_dim, self.attention_dim),
                                  initializer=RandomNormal(),
                                  trainable=True,
                                  name="Ua")
        self.Va = self.add_weight(shape=(self.attention_dim,),
                                  initializer=Zeros(),
                                  trainable=True,
                                  name="Va")

        if self.use_scale:
            self.scale = self.add_weight(
                shape=(),
                initializer="ones",
                trainable=True,
                name="scale"
            )

        super(OpinionsEmbedding, self).build(input_shape)

    def call(self, inputs, mask=None):
        """
        Forward pass for the attention layer.
        
        Args:
            inputs: Input tensor of shape (batch_size, seq_len, features).
            mask: Optional mask tensor of shape (batch_size, seq_len).
        
        Returns:
            context_vector: Tensor of shape (batch_size, features).
        """
        context_vectors = []
        for i in range(self.num_heads):
            Q = self.query_dense_layers[i](inputs)
            K = self.key_dense_layers[i](inputs)
            V = self.value_dense_layers[i](inputs)

            score = tf.tanh(tf.tensordot(Q, self.Wa, axes=[2, 0]) + tf.tensordot(K, self.Ua, axes=[2, 0]) + self.Va)

            if self.use_scale:
                score = score * self.scale
            
            score = tf.nn.softmax(score, axis=1)

            if mask is not None:
                score *= tf.cast(mask[:, tf.newaxis, :], dtype=score.dtype)
            
            context_vector = tf.reduce_sum(score * V, axis=1)
            context_vectors.append(context_vector)

        output = tf.concat(context_vectors, axis=-1)
        
        return output

    def get_config(self):
        config = super(OpinionsEmbedding, self).get_config()
        config.update({
            "attention_dim": self.attention_dim,
            "num_heads": self.num_heads,
            "use_scale": self.use_scale
        })
        return config

# TransformerBock
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, num_layers=2, rate=0.1):
        """
        Args:
            embed_dim: Dimension of the embedding space.
            num_heads: Number of attention heads.
            ff_dim: Dimensionality of the feed-forward network.
            num_layers: Number of transformer layers (depth).
            rate: Dropout rate.
        """
        super(TransformerBlock, self).__init__()
        self.num_layers = num_layers
        
        self.attention_layers = [MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim) for _ in range(num_layers)]
        self.ffn_layers = [tf.keras.Sequential([
            Dense(ff_dim, activation="relu"), 
            Dense(embed_dim)
        ]) for _ in range(num_layers)]
        self.layernorm1_layers = [LayerNormalization(epsilon=1e-6) for _ in range(num_layers)]
        self.layernorm2_layers = [LayerNormalization(epsilon=1e-6) for _ in range(num_layers)]
        self.dropout1_layers = [Dropout(rate) for _ in range(num_layers)]
        self.dropout2_layers = [Dropout(rate) for _ in range(num_layers)]

    def call(self, inputs, training):
        x = inputs
        for i in range(self.num_layers):
            attn_output = self.attention_layers[i](x, x)
            attn_output = self.dropout1_layers[i](attn_output, training=training)
            x = self.layernorm1_layers[i](x + attn_output)
            
            ffn_output = self.ffn_layers[i](x)
            ffn_output = self.dropout2_layers[i](ffn_output, training=training)
            x = self.layernorm2_layers[i](x + ffn_output)
        
        return x

maxlen = 100
vocab_size = tokenizer.vocab_size + 1  
opinions_vocab_size = len(label_encoder_opinions.classes_) 
embedding_dim = 128
num_heads = 4
ff_dim = 128
embed_dim = 128


# Define MTOT
input_ids = Input(shape=(maxlen,), dtype=tf.int32, name="input_ids")
attention_mask = Input(shape=(maxlen,), dtype=tf.int32, name="attention_mask")
opinions_input = Input(shape=(1,), dtype=tf.int32, name="opinions_input")

embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
x = embedding_layer(input_ids)

cnn_out = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(x)
cnn_out = MaxPooling1D(pool_size=2, padding='same')(cnn_out)

desired_length = maxlen // 2 
cnn_out = Conv1D(filters=embed_dim, kernel_size=1, activation='relu')(cnn_out)
cnn_out = tf.keras.layers.Lambda(lambda x: x[:, :desired_length])(cnn_out) 

bigru_out = Bidirectional(GRU(64, return_sequences=True))(x)
bigru_out = tf.keras.layers.Lambda(lambda x: x[:, :desired_length])(bigru_out) 

concat_out = Concatenate()([cnn_out, bigru_out])

# Transformer blocks
num_layers = 4  
x = concat_out
for _ in range(num_layers):
    x = TransformerBlock(embed_dim * 2, num_heads=8, ff_dim=128)(x)

x = tf.keras.layers.GlobalAveragePooling1D()(x)
x = Dropout(0.1)(x)
x = LayerNormalization(epsilon=1e-6)(x)

opinions_embed = Embedding(input_dim=opinions_vocab_size, output_dim=embedding_dim)(opinions_input)
opinions_embed = OpinionsEmbedding(attention_dim=128)(opinions_embed)

combined = tf.keras.layers.concatenate([x, opinions_embed])

status_output = Dense(status_labels.shape[1], activation='softmax', name='status_output')(combined)
sentiment_output = Dense(sentiment_labels.shape[1], activation='softmax', name='sentiment_output')(combined)

model = tf.keras.Model(inputs=[input_ids, attention_mask, opinions_input], outputs=[status_output, sentiment_output])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
    loss={'status_output': 'categorical_crossentropy', 'sentiment_output': 'categorical_crossentropy'},
    metrics={'status_output': 'accuracy', 'sentiment_output': 'accuracy'}
)

model.summary()

# Define Early Stopping
early_stopping = EarlyStopping(
    monitor='val_loss',    
    patience=4,            
    restore_best_weights=True,  
    verbose=1              
)

# Training the Model
history = model.fit(
    x={
        'input_ids': X_train,
        'attention_mask': attention_train,
        'opinions_input': y_opinions_train
    },
    y={
        'status_output': y_status_train,
        'sentiment_output': y_sentiment_train
    },
    validation_data=(
        {
            'input_ids': X_val,
            'attention_mask': attention_val,
            'opinions_input': y_opinions_val
        },
        {
            'status_output': y_status_val,
            'sentiment_output': y_sentiment_val
        }
    ),
    epochs=30,
    batch_size=16,
    callbacks=[early_stopping]
)

# Ensure test data is prepared with opinions input
X_test = {
    'input_ids': X_test,
    'attention_mask': attention_test,
    'opinions_input': y_opinions_test
}

y_test = {
    'status_output': y_status_test,
    'sentiment_output': y_sentiment_test
}

# Evaluate the model
evaluation_results = model.evaluate(
    x=X_test,
    y=y_test,
    verbose=1
)

# Print evaluation results
print("Evaluation results:")
for metric_name, metric_value in zip(model.metrics_names, evaluation_results):
    print(f"{metric_name}: {metric_value:.4f}")

# Get predictions
predictions = model.predict(X_test)

# Extract predictions for each output
status_predictions = predictions[0]
sentiment_predictions = predictions[1]

status_pred_labels = np.argmax(status_predictions, axis=1)
sentiment_pred_labels = np.argmax(sentiment_predictions, axis=1)

y_status_test_labels = np.argmax(y_status_test, axis=1)
y_sentiment_test_labels = np.argmax(y_sentiment_test, axis=1)

# Generate classification report for 'status'
status_report = classification_report(y_status_test_labels, status_pred_labels, target_names=label_encoder_status.classes_)
print("Classification Report for Status:\n", status_report)

# Generate classification report for 'sentiment'
sentiment_report = classification_report(y_sentiment_test_labels, sentiment_pred_labels, target_names=label_encoder_sentiment.classes_)
print("Classification Report for Sentiment:\n", sentiment_report)

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting normalize=True.
    """
    if not title:
        title = 'Normalized Confusion Matrix' if normalize else 'Confusion Matrix'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Normalize the confusion matrix if required
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    # Plotting the confusion matrix
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    
    # Set the ticks and labels
    ax.set(xticks=np.arange(len(classes)),
           yticks=np.arange(len(classes)),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True Label',
           xlabel='Predicted Label')

    # Rotate the tick labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Annotate the confusion matrix with values
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()
    plt.show()

# Example usage:
plot_confusion_matrix(y_status_test_labels, status_pred_labels, classes=label_encoder_status.classes_)

plot_confusion_matrix(y_sentiment_test_labels, sentiment_pred_labels, classes=label_encoder_sentiment.classes_)

# Binarize the labels for ROC curve calculation
y_status_test_bin = label_binarize(y_status_test_labels, classes=np.arange(status_labels.shape[1]))
y_sentiment_test_bin = label_binarize(y_sentiment_test_labels, classes=np.arange(sentiment_labels.shape[1]))

# Compute ROC curve and AUC for each class in the 'status' task
status_fpr, status_tpr, _ = roc_curve(y_status_test_bin.ravel(), status_predictions.ravel())
status_auc = auc(status_fpr, status_tpr)

# Compute ROC curve and AUC for each class in the 'sentiment' task
sentiment_fpr, sentiment_tpr, _ = roc_curve(y_sentiment_test_bin.ravel(), sentiment_predictions.ravel())
sentiment_auc = auc(sentiment_fpr, sentiment_tpr)


# Define a function to plot class-wise ROC curves
def plot_class_wise_roc(y_true_bin, y_pred, class_names, title):
    n_classes = y_true_bin.shape[1]
    
    plt.figure(figsize=(5, 5))
    
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'Class {class_names[i]} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.title(title)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.show()

# Plot class-wise ROC curve for 'status'
plot_class_wise_roc(y_status_test_bin, status_predictions, label_encoder_status.classes_, 'ROC Curve for Status')

# Plot class-wise ROC curve for 'sentiment'
plot_class_wise_roc(y_sentiment_test_bin, sentiment_predictions, label_encoder_sentiment.classes_, 'ROC Curve for Sentiment')

opinion_embedding_model = tf.keras.Model(inputs=model.input, outputs=model.get_layer('embedding_5').output)

opinion_indices = np.array([[i] for i in range(opinions_vocab_size)])

dummy_input_ids = np.zeros((opinions_vocab_size, maxlen))  
dummy_attention_mask = np.zeros((opinions_vocab_size, maxlen))  


opinion_embeddings = opinion_embedding_model.predict(
    [dummy_input_ids, dummy_attention_mask, opinion_indices]
)

n_clusters = 5  

kmeans = KMeans(n_clusters=n_clusters, random_state=0)
clusters = kmeans.fit_predict(opinion_embeddings.squeeze())

pca = PCA(n_components=2)
reduced_embeddings = pca.fit_transform(opinion_embeddings.squeeze())

# Plotting
plt.figure(figsize=(10, 7))
plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=clusters, cmap='viridis', edgecolors='k', alpha=0.7)
plt.title('Opinion Embeddings Clustering')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid(True)
plt.show()

positional_embeddings = embedding_layer._create_sinusoidal_embeddings(maxlen, embed_dim)

plt.figure(figsize=(10, 8))
plt.pcolormesh(positional_embeddings[0], cmap='viridis')
plt.xlabel('Embedding Dimension')
plt.ylabel('Position')
plt.colorbar(label='Positional Encoding Value')
plt.title('Sinusoidal Positional Embeddings')
plt.show()




























































































