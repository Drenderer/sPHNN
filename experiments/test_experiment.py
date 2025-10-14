from model_ingredient import get_model, model_ingredient, sphnn
from sacred import Experiment, cli_option
from sacred.observers import SqlObserver

# DB_PATH = Path(__file__).parent / "experiment.db"


@cli_option("-T", "--tag")
def parse_tags(args, run):
    run.info["tags"] = [t.strip() for t in args.split(",") if t.strip()]


test_experiment = Experiment(
    "test_experiment",
    ingredients=(model_ingredient, ),
    additional_cli_options=[parse_tags],
)
# snippet_training.observers.append(SqlObserver(f"sqlite:///{DB_PATH}"))


# test_experiment.named_config(sphnn)

@test_experiment.named_config
def some_config():
    help_me = True

@test_experiment.automain
def main(model):
    model = get_model()
    print("This is a test experiment.")