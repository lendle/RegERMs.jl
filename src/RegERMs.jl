module RegERMs
using Optim

export RegERM, optimize

abstract RegERM

# Pretty-print
function Base.show(io::IO, model::RegERM)
	println(io, "$(modelname(model))")
	println(io, repeat("-", length(modelname(model))))
	println(io, "regularization parameter: $(model.λ)")
	println(io, "number of examples:       $(model.num_examples)")
	println(io, "number of features:       $(model.num_features)")
end

function optimize(model::RegERM)
	# start value
	w0 = [1.0*x for x in zeros(model.num_features,1)]

	obj(w::Vector) = sum(losses(model, w)[1]) + regularizer(model, w)[1]
	grad(w::Vector) = sum(losses(model, w)[2]) + regularizer(model, w)[2]

	Optim.optimize(obj, w0, method=:l_bfgs).minimum
end

include("loss.jl")
include("regularizer.jl")
include("svm.jl")
include("logreg.jl")

end # module