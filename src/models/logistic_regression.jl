## binomial

immutable BinomialLogReg <: RegERM
    X::Matrix               # n x m matrix of n m-dimensional training examples
    y::Vector               # 1 x n vector with training classes
    n::Int                  # number of training examples
    m::Int                  # number of features
    kernel::Symbol          # kernel function
    regression_type::Symbol # ordinal, binomial, multinomial
end

function BinomialLogReg(X::Matrix, y::Vector; kernel::Symbol=:linear)
    check_arguments(X, y)
    BinomialLogReg(X, y, size(X)..., kernel, :binomial)
end

methodname(::BinomialLogReg) = "Logistic Regression"
loss(::BinomialLogReg) = LogisticLoss()
regularizer(::BinomialLogReg, w::Vector, λ::Float64) = L2reg(w, λ)

## multinomial

immutable MultinomialLogReg <: RegERM
    X::Matrix               # n x m matrix of n m-dimensional training examples
    y::Vector               # 1 x n vector with training classes
    n::Int                  # number of training examples
    m::Int                  # number of features
    kernel::Symbol          # kernel function
    regression_type::Symbol # ordinal, binomial, multinomial
end

function MultinomialLogReg(X::Matrix, y::Vector; kernel::Symbol=:linear)
    #check_arguments(X, y)
    MultinomialLogReg(X, y, size(X)..., kernel, :multinomial)
end

methodname(::MultinomialLogReg) = "Logistic Regression"
loss(::MultinomialLogReg) = MultinomialLogisticLoss()
regularizer(::MultinomialLogReg, w::Vector, λ::Float64) = L2reg(w, λ)