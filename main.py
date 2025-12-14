# ============================================
# Step 1: Import Required Libraries
# ============================================

from pyspark.sql import SparkSession
from pyspark.sql.functions import rand
from pyspark.ml.feature import VectorAssembler
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# ============================================
# Step 2: Initialize Spark Session
# ============================================

spark = SparkSession.builder \
    .appName("BinaryClassificationNN") \
    .getOrCreate()

# ============================================
# Step 3: Create Synthetic Binary Classification Data
# ============================================
# Features: x1, x2, x3, x4
# Label Rule (Realistic):
# If weighted sum > threshold → Class 1 else Class 0

num_samples = 1000

data = spark.range(0, num_samples)

data = data.withColumn("x1", rand()) \
           .withColumn("x2", rand()) \
           .withColumn("x3", rand()) \
           .withColumn("x4", rand())

# Create label based on non-linear condition
data = data.withColumn(
    "label",
    ((data.x1 * 0.3 + data.x2 * 0.4 + data.x3 * 0.2 + data.x4 * 0.1) > 0.5).cast("int")
)

data.show(5)

# ============================================
# Step 4: Feature Vectorization (PySpark)
# ============================================

assembler = VectorAssembler(
    inputCols=["x1", "x2", "x3", "x4"],
    outputCol="features"
)

final_data = assembler.transform(data).select("features", "label")

# ============================================
# Step 5: Convert Spark DataFrame to NumPy
# ============================================

rows = final_data.collect()

X = np.array([row["features"] for row in rows])
y = np.array([row["label"] for row in rows])

# ============================================
# Step 6: Build Feedforward Neural Network
# ============================================

model = Sequential()

# Input + Hidden Layer 1
model.add(Dense(64, activation='relu', input_shape=(X.shape[1],)))

# Hidden Layer 2
model.add(Dense(32, activation='relu'))

# Output Layer
model.add(Dense(1, activation='sigmoid'))

# ============================================
# Step 7: Compile the Model
# ============================================

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# ============================================
# Step 8: Train the Model
# ============================================

model.fit(
    X,
    y,
    epochs=20,
    batch_size=32,
    validation_split=0.2
)

# ============================================
# Step 9: Evaluate the Model
# ============================================

loss, accuracy = model.evaluate(X, y)
print("Final Model Accuracy:", accuracy)

# ============================================
# Step 10: Stop Spark Session
# ============================================

spark.stop()
