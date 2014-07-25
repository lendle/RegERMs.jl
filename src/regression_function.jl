
abstract RegressionFunction

# Linear predictive function

type LinearRegressionFunction <: RegressionFunction
end

values(f::LinearRegressionFunction, X::AbstractMatrix, w::Vector) = X*w
gradient(f::LinearRegressionFunction, X::AbstractMatrix, w::Vector) = X


# TODO: Dual predictive function