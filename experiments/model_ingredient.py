import jax
from dynax import ISPHS, ConvexLyapunov, ODESolver
from jax import random as jr
from jax.nn.initializers import glorot_uniform, zeros
from klax.nn import FICNN, ConstantSPDMatrix, SkewSymmetricMatrix
from sacred import Ingredient

from sphnn.models.neuralode import NODE
from sphnn.models.normalization import Normalization, NormalizationWrapper

INITIALIZERS = dict(glorot_uniform=glorot_uniform(), zeros=zeros)

model_ingredient = Ingredient("model")


@model_ingredient.config
def default():
    state_size = 8
    input_size = 4
    wrap_with = "nothing"
    augmentation = 0
    max_steps = 4096


@model_ingredient.named_config
def node():
    name = "NODE"
    specifics = dict(
        key_seed=0,
        width_sizes=[16, 16],
        weight_init="glorot_uniform",
        bias_init="zeros",
    )


@model_ingredient.named_config
def sphnn():
    name = "sPHNN"
    specifics = dict(
        key_seed=0,
        structure_matrix_width_sizes=[16, 16],
        structure_matrix_weight_init="glorot_uniform",
        resistive_matrix_init="zeros",
        resistive_matrix_epsilon=1e-6,
        ficnn_width_sizes=[16, 16],
        ficnn_weight_init="glorot_uniform",
        ficnn_bias_init="zeros",
    )


def get_sphnn(options: dict):
    model_key = jax.random.key(options["key_seed"])
    j_key, r_key, ficnn_key, h_key = jr.split(model_key, 4)

    state_size = options["state_size"]

    structure_matrix = SkewSymmetricMatrix(
        state_size,
        (state_size, state_size),
        width_sizes=options["structure_matrix_width_sizes"],
        weight_init=INITIALIZERS[options["structure_matrix_weight_init"]],
        key=j_key,
    )
    resistive_matrix = ConstantSPDMatrix(
        (state_size, state_size),
        epsilon=options["resistive_matrix_epsilon"],
        init=INITIALIZERS[options["resistive_matrix_init"]],
        key=r_key,
    )
    ficnn = FICNN(
        state_size,
        "scalar",
        width_sizes=options["ficnn_width_sizes"],
        weight_init=INITIALIZERS[options["ficnn_weight_init"]],
        bias_init=INITIALIZERS[options["ficnn_bias_init"]],
        key=ficnn_key,
    )
    hamiltonian = ConvexLyapunov(
        ficnn,
        state_size,
        minimum_init=zeros,
        minimum_learnable=False,
        epsilon=options["hamiltonian_epsilon"],
        key=h_key,
    )
    return ISPHS(hamiltonian, structure_matrix, resistive_matrix)


def get_node(options: dict):
    model_key = jax.random.key(options["key_seed"])
    node = NODE(
        state_size=options["state_size"],
        input_size=options["input_size"],
        width_sizes=options["width_sizes"],
        weight_init=INITIALIZERS[options["weight_init"]],
        bias_init=INITIALIZERS[options["bias_init"]],
        key=model_key,
    )
    return node


@model_ingredient.capture
def get_model(name, state_size, input_size, wrap_with, augmentation, max_steps, specifics):

    options = dict(state_size=state_size, input_size=input_size)
    options.update(specifics)

    if name == "NODE":
        model = get_node(options)
    else:
        raise ValueError(f"Unknown model name {name}.")

    model = ODESolver(model, augmentation, max_steps)

    if wrap_with == "normalizer":
        normalizer = Normalization.init_identity(state_size, input_size)
        return NormalizationWrapper(model, normalizer)
    else:
        return model
