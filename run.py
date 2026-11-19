import hydra


@hydra.main(config_path="config", config_name="unfold_base", version_base=None)
def run(cfg):
    if cfg.task == "unfolding":
        from src.unfolding.main import main

        main(cfg)
    elif cfg.task == "classification":
        from src.classification.main import main

        main(cfg)
    else:
        raise ValueError(f"Unknown task: {cfg.task}")


if __name__ == "__main__":
    run()
