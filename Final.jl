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

#### Risk Model w/o attention

## First training the model
data_path = "Data/All estimation raw data.csv"
data = CSV.read(data_path, DataFrame)

data.Location = ifelse.(data.Location .== "Technion", 1, 0)
data.Gender = ifelse.(data.Gender .== "M", 1, 0)
data.Button = ifelse.(data.Button .== "R", 1, 0)

LotShapeABool = select(data, [:LotShapeA => ByRow(isequal(v))=> Symbol(v) for v in unique(data.LotShapeA)])
rename!(LotShapeABool, [:RskewA, :NoneA, :SymmA, :LskewA])
data = hcat(data, LotShapeABool)

LotShapeBBool = select(data, [:LotShapeB => ByRow(isequal(v))=> Symbol(v) for v in unique(data.LotShapeB)])
rename!(LotShapeBBool, [:NoneB, :SymmB, :RSkewB, :LskewB])
data = hcat(data, LotShapeBBool)

features = select(data, [:Location, :Age, :Gender, :Apay, :Bpay, :Ha, :pHa, :La, :Hb, :pHb, :Lb, :Forgone, 
    :RskewA, :NoneA, :SymmA, :LskewA, :Amb])
features = Matrix{Float32}(features)
features = Flux.normalise(features, dims=2)
features = transpose(features) # Need to do this so dimensions match

labels = Flux.onehotbatch(data.B, [false, true])

function earlystoptrain(epochs, model, loss_threshold, patience, best_loss, epochs_since_improvement,  
    train_features, train_labels, test_features, test_labels, best_model)
    for epoch in 1:epochs
        Flux.train!(loss, Flux.params(model), [(train_features, train_labels)], optimizer)
        val_loss = loss(test_features, test_labels)
        println("Validation Loss: $val_loss")
        if val_loss < loss_threshold
            println("Loss threshold reached at epoch $epoch with validation loss $val_loss")
            best_model = deepcopy(model)
            break
        end
        if val_loss < best_loss
            best_loss = val_loss
            epochs_since_improvement = 0
            best_model = deepcopy(model)
            println("New best model saved with loss $best_loss at epoch $epoch")
        else
            epochs_since_improvement += 1
        end
        if epochs_since_improvement >= patience
            println("Stopping early after $epochs_since_improvement epochs without improvement.")
            break
        end
    end
    return best_model 
end

(train_features, train_labels), (test_features, test_labels) = splitobs((features, labels), at=0.8)

risk_model = Chain(
    Dense(size(train_features, 1), 16, relu),
    Dense(16, 16, relu),
    Dense(16, 16, relu),
    Dense(16, 2), 
    softmax
)

loss(x, y) = Flux.crossentropy(risk_model(x), y)
optimizer = ADAM(0.001)

best_model = deepcopy(risk_model)
best_risk_model = earlystoptrain(1000, risk_model, 0.01, 10, Inf, 0, train_features, train_labels, test_features, test_labels)

accuracy(x, y) = mean(Flux.onecold(best_risk_model(x)) .== Flux.onecold(y))
println("Test set accuracy: $(accuracy(test_features, test_labels))")

predictions = Flux.onecold(best_risk_model(test_features)) 

cm = confusmat(2, Flux.onecold(test_labels), predictions)  

println("Confusion Matrix:\n", cm) # [55173 187; 3084 43706]


## Predicting the test dataset
new_data_path = "Data/raw-comp-set-data-Track-2.csv"
new_data = CSV.read(new_data_path, DataFrame)

new_data.Location = ifelse.(new_data.Location .== "Technion", 1, 0)
new_data.Gender = ifelse.(new_data.Gender .== "M", 1, 0)
new_data.Button = ifelse.(new_data.Button .== "R", 1, 0)

testLotShapeABool = select(new_data, [:LotShapeA => ByRow(isequal(v))=> Symbol(v) for v in unique(data.LotShapeA)])
rename!(testLotShapeABool, [:RskewA, :NoneA, :SymmA, :LskewA])
new_data = hcat(new_data, testLotShapeABool)

testLotShapeBBool = select(new_data, [:LotShapeB => ByRow(isequal(v))=> Symbol(v) for v in unique(data.LotShapeB)])
rename!(LotShapeBBool, [:NoneB, :SymmB, :RSkewB, :LskewB])
new_data = hcat(new_data, testLotShapeBBool)

new_features = select(new_data, [:Location, :Age, :Gender, :Apay, :Bpay, :Ha, :pHa, :La, :Hb, :pHb, :Lb, :Forgone, 
:RskewA, :NoneA, :SymmA, :LskewA, :Amb])
new_features = Matrix{Float32}(new_features)  
new_features = Flux.normalise(new_features, dims=2)
new_features = transpose(new_features)

new_predictions = Flux.onecold(best_risk_model(new_features))  
actual_labels = new_data.B

new_cm = confusmat(2, actual_labels .+ 1, new_predictions)
println("Confusion Matrix:\n", new_cm)

new_cmAccuracy = sum(diag(new_cm)) / sum(new_cm) # [1893 5; 33 1819]
println("Test set accuracy: $new_cmAccuracy") # 0.98987

#### Now incorporating attention elements

# Need order, trial
attn_features = select(data, [:Location, :Age, :Gender, :Apay, :Bpay, :Ha, :pHa, :La, :Hb, :pHb, :Lb, :Forgone, 
    :RskewA, :NoneA, :SymmA, :LskewA, :Amb, :Order, :Trial])
attn_features = Matrix{Float32}(attn_features)  
attn_features = Flux.normalise(attn_features, dims=2)
attn_features = transpose(attn_features) 

labels = Flux.onehotbatch(data.B, [false, true])

(train_attn_features, train_labels), (test_attn_features, test_labels) = splitobs((attn_features, labels), at=0.8)

attn_model = Chain(
    Dense(size(train_attn_features, 1), 16, relu),
    Dense(16, 16, relu),
    Dense(16, 16, relu),
    Dense(16, 2),
    softmax
)

loss(x, y) = Flux.crossentropy(attn_model(x), y)
optimizer = ADAM(0.001)

best_model = deepcopy(attn_model)
best_attn_model = earlystoptrain(1000, attn_model, 0.01, 20, Inf, 0,
train_attn_features, train_labels, test_attn_features, test_labels, best_model)

accuracy(x, y) = mean(Flux.onecold(best_attn_model(x)) .== Flux.onecold(y))
println("Test set accuracy: $(accuracy(test_attn_features, test_labels))")

predictions = Flux.onecold(best_attn_model(test_attn_features)) 

cm = confusmat(2, Flux.onecold(test_labels), predictions)  

println("Confusion Matrix:\n", cm) # [55173 187; 3084 43706]


## Predicting the test dataset
attn_new_features = select(new_data, [:Location, :Age, :Gender, :Apay, :Bpay, :Ha, :pHa, :La, :Hb, :pHb, :Lb, :Forgone, 
:RskewA, :NoneA, :SymmA, :LskewA, :Amb, :Order, :Trial])
attn_new_features = Matrix{Float32}(attn_new_features)  
attn_new_features = Flux.normalise(attn_new_features, dims=2)
attn_new_features = transpose(attn_new_features)

new_predictions = Flux.onecold(best_attn_model(attn_new_features))  
actual_labels = new_data.B

attn_new_cm = confusmat(2, actual_labels .+ 1, new_predictions) # [1895 3; 36 1816]
println("Confusion Matrix:\n", attn_new_cm)

attn_new_cmAccuracy = sum(diag(attn_new_cm)) / sum(attn_new_cm) # 0.9896
println("Test set accuracy: $attn_new_cmAccuracy")