using CSV
using DataFrames
using Flux
using MLDataUtils
using Statistics
using LinearAlgebra
using MLBase
using StatsBase

# Load data
data_path = "All estimation raw data.csv"
data = filter(row -> row.RT != "NA", CSV.read(data_path, DataFrame))

# Preprocess features: Gender encoding, select relevant columns
data.Gender = ifelse.(data.Gender .== "M", 1, 0)
data.Button = ifelse.(data.Button .== "R", 1, 0)

LotShapeABool = select(data, [:LotShapeA => ByRow(isequal(v))=> Symbol(v) for v in unique(data.LotShapeA)])
rename!(LotShapeABool, [:RskewA, :NoneA, :SymmA, :LskewA])
data = hcat(data, LotShapeABool)

LotShapeBBool = select(data, [:LotShapeB => ByRow(isequal(v))=> Symbol(v) for v in unique(data.LotShapeB)])
rename!(LotShapeBBool, [:NoneB, :SymmB, :RSkewB, :LskewB])
data = hcat(data, LotShapeBBool)

features = select(data, [:Age, :Gender, :Apay, :Bpay, :Ha, :pHa, :La, :Hb, :pHb, :Lb, :Forgone, 
    :RskewA, :NoneA, :SymmA, :LskewA])
features = Matrix{Float32}(features)  # Convert DataFrame to Matrix

# Normalize features
normed_features = Flux.normalise(features, dims=2)

# Transpose features so that each column is an observation
features = transpose(features)  # Convert from 510750 x 12 to 12 x 510750

# Prepare labels for classification
labels = Flux.onehotbatch(data.B, [false, true])

# Split data
(train_features, train_labels), (test_features, test_labels) = splitobs((features, labels), at=0.8)

model = Chain(
    Dense(size(train_features, 1), 64, relu),
    Dense(64, 64, relu),
    Dense(64, 64, relu),
    Dense(64, 32, relu),
    Dense(32, 2),  # Two outputs for two classes
    softmax
)

loss(x, y) = Flux.crossentropy(model(x), y)
optimizer = ADAM(0.001)

epochs = 300
for epoch in 1:epochs
    Flux.train!(loss, Flux.params(model), [(train_features, train_labels)], optimizer)
    println("Epoch $epoch: Loss $(loss(train_features, train_labels))")
end

accuracy(x, y) = mean(Flux.onecold(model(x)) .== Flux.onecold(y))
println("Test set accuracy: $(accuracy(test_features, test_labels))")

# Generate predictions
predictions = Flux.onecold(model(test_features))  # Adjust indices if necessary

# Create confusion matrix
cm = confusmat(2, Flux.onecold(test_labels), predictions)  # Adjust indices for true labels as well

# Display confusion matrix
println("Confusion Matrix:")
println(cm)
    
# Calculate accuracy
cmAccuracy = sum(diag(cm)) / sum(cm)
println("Test set accuracy: $accuracy")





# Load the new dataset
new_data_path = "raw-comp-set-data-Track-2.csv"
new_data = CSV.read(new_data_path, DataFrame)

# Preprocess features: assuming the same preprocessing as the training data
new_data.Gender = ifelse.(new_data.Gender .== "M", 1, 0)
new_data.Button = ifelse.(new_data.Button .== "R", 1, 0)
new_features = select(new_data, [:Age, :Gender, :Ha, :pHa, :La, :LotNumA, :LotNumB, :Hb, :pHb, :Lb, :Amb, 
:Corr, :Order, :Payoff, :Forgone, :Trial, :Button])
new_features = Matrix{Float32}(new_features)  # Convert DataFrame to Matrix

# Normalize features
for i in 1:size(new_features, 2)
    new_features[:, i] .= (new_features[:, i] .- mean(new_features[:, i])) ./ std(new_features[:, i])
end

# Transpose features so that each column is an observation
new_features = transpose(new_features)  # Convert from rows to columns

# Use the model to predict new data
predictions = model(new_features)

# If the last layer of your model uses softmax (as it should for classification tasks),
# `predictions` will give you the probabilities of each class for each observation
predicted_classes = Flux.onecold(predictions)  # Convert probabilities to class labels

# Map numeric classes back to boolean or actual class names if necessary
class_names = [false, true]  # Depending on how you encoded labels initially
predicted_labels = Int.(class_names[predicted_classes])

actual_labels = new_data.B
test = mean(predicted_labels .== actual_labels)
println("Accuracy: ", test)

# Create confusion matrix
predicted_labels_conf = predicted_labels .+ 1  # Convert 0-based to 1-based indexing
actual_labels_conf = actual_labels .+ 1 
conf_matrix = confusmat(2, actual_labels_conf, predicted_labels_conf)  # 2 indicates binary classification 
println("Confusion Matrix:\n", conf_matrix)

results_data = DataFrame(
    ID = new_data.SubjID,  # Assuming there's an 'ID' column in your original data
    PredictedLabels = predicted_labels,
    ActualLabels = actual_labels,
    Comparison = predicted_labels .== actual_labels
)

# Write the DataFrame with results to a new CSV file
CSV.write("test.csv", results_data)