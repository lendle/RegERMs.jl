export SVM

immutable SVM <: RegERM
    X::Matrix           # n x m matrix of n m-dimensional training examples
    y::Vector           # 1 x n vector with training classes
    λ::Float64          # regularization parameter
    num_features::Int   # number of features
    num_examples::Int   # number of training examples
end

function SVM(X::Matrix, y::Vector, λ::Float64)
	(n, m) = size(X)
	if (n != length(y))
		error("dimension mismatch. Try: X'")
	end
	SVM(X, y, λ, m, n)
end

modelname(svm::SVM) = "Support Vector Machine"
losses{T<:Real}(svm::SVM, w::Vector{T}) = hinge(svm, w)
regularizer{T<:Real}(svm::SVM, w::Vector{T}) = l2reg(w, svm.λ)