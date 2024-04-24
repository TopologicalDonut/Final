### Computational Econ Final 
### Author: Stephen Min

using CSV
using DataFrames
using Flux
using MLDataUtils
using Statistics
using LinearAlgebra
using MLBase
using StatsBase

## First training the model
data_path = "All estimation raw data.csv"
data = filter(row -> row.RT != "NA", CSV.read(data_path, DataFrame))

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
features = Flux.normalise(features, dims=2)
features = transpose(features) # Need to do this so dimensions match

labels = Flux.onehotbatch(data.B, [false, true])

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

epochs = 250
for epoch in 1:epochs
    Flux.train!(loss, Flux.params(model), [(train_features, train_labels)], optimizer)
    println("Epoch $epoch: Loss $(loss(train_features, train_labels))")
end

accuracy(x, y) = mean(Flux.onecold(model(x)) .== Flux.onecold(y))
println("Test set accuracy: $(accuracy(test_features, test_labels))")

predictions = Flux.onecold(model(test_features)) 

cm = confusmat(2, Flux.onecold(test_labels), predictions)  

println("Confusion Matrix:\n", cm) # [19694 72; 22 15462]
    
cmAccuracy = sum(diag(cm)) / sum(cm)
println("Test set accuracy: $cmAccuracy") # 0.9973333333333333


## Predicting the test dataset
new_data_path = "raw-comp-set-data-Track-2.csv"
new_data = CSV.read(new_data_path, DataFrame)

new_data.Gender = ifelse.(new_data.Gender .== "M", 1, 0)
new_data.Button = ifelse.(new_data.Button .== "R", 1, 0)

testLotShapeABool = select(new_data, [:LotShapeA => ByRow(isequal(v))=> Symbol(v) for v in unique(data.LotShapeA)])
rename!(testLotShapeABool, [:RskewA, :NoneA, :SymmA, :LskewA])
new_data = hcat(new_data, testLotShapeABool)

testLotShapeBBool = select(new_data, [:LotShapeB => ByRow(isequal(v))=> Symbol(v) for v in unique(data.LotShapeB)])
rename!(LotShapeBBool, [:NoneB, :SymmB, :RSkewB, :LskewB])
new_data = hcat(new_data, testLotShapeBBool)

new_features = select(new_data, [:Age, :Gender, :Apay, :Bpay, :Ha, :pHa, :La, :Hb, :pHb, :Lb, :Forgone, 
:RskewA, :NoneA, :SymmA, :LskewA])
new_features = Matrix{Float32}(new_features)  
new_features = Flux.normalise(new_features, dims=2)
new_features = transpose(new_features)

new_predictions = Flux.onecold(model(new_features))  
actual_labels = new_data.B

new_cm = confusmat(2, actual_labels .+ 1, new_predictions) # [1896 2; 11 1841]
println("Confusion Matrix:\n", new_cm)

new_cmAccuracy = sum(diag(new_cm)) / sum(new_cm) # 0.9965333333333334
println("Test set accuracy: $new_cmAccuracy")
