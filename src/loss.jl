
abstract Loss

# for discriminative (conditional) models
abstract OrdinalLoss <: Loss
abstract NominalLoss <: Loss

# for binary-class cases
abstract BinomialLoss <: NominalLoss
# for multi-class cases
abstract MultinomialLoss <: NominalLoss

# fall back
value_and_deriv(l::Loss, fv::Real, y::Real) = (value(l, fv, y), deriv(l, fv, y))

function tloss(l::Loss, fv::AbstractVector, y::AbstractVector)
    n = size(fv, 1)  # n is the number of samples
    s = 0.0
    for i = 1:n
        s += value(l, fv[i], y[i])
    end
    return s
end

function values(l::Loss, fv::AbstractVector, y::AbstractVector)
    n = size(fv, 1)  # n is the number of samples
    v = zeros(n)
    for i = 1:n
        v[i] = value(l, fv[i], y[i])
    end
    return v
end

function derivs(l::Loss, fv::AbstractVector, y::AbstractVector)
    n = size(fv, 1)  # n is the number of samples
    dv = zeros(n)
    for i = 1:n
        dv[i] = deriv(l, fv[i], y[i])
    end
    return dv
end

## logistic loss

type LogisticLoss <: BinomialLoss end

value(l::LogisticLoss, fv::Real, y::Int) = if fv>34 -y*fv else log(1+exp(-y*fv)) end
deriv(l::LogisticLoss, fv::Real, y::Int) = -y / (1 + exp(y*fv))

## squared loss

type SquaredLoss <: OrdinalLoss end

value(l::SquaredLoss, fv::Real, y::Real) = 0.5 * (fv-y) * (fv-y)
deriv(l::SquaredLoss, fv::Real, y::Real) = fv-y

## hinge loss

type HingeLoss <: BinomialLoss end

value(l::HingeLoss, fv::Real, y::Int) = max(0, 1-y*fv)
deriv(l::HingeLoss, fv::Real, y::Int) = -y*(value(l, fv, y) > 0)

value_and_deriv(l::HingeLoss, fv::Real, y::Int) = (h = max(0, 1-y*fv); (h, -y*(h>0)))