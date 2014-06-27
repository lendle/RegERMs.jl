export LogReg

immutable LogReg <: RegERM
    X::Matrix  # n x m matrix of n m-dimensional training examples
    y::Vector  # 1 x n vector with training classes
    n::Int     # number of training examples
    m::Int     # number of features
end

function LogReg(X::Matrix, y::Vector)
	check_arguments(X, y)
	LogReg(X, y, size(X)...)
end

modelname(::LogReg) = "Logistic Regression"
loss(::LogReg, w::Vector, X::Matrix, y::Vector) = Logistic(w, X, y)
regularizer(::LogReg, w::Vector, λ::Float64) = L2reg(w, λ)