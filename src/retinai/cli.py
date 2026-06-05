from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from retinai.config import ensure_artifact_dirs, load_config
from retinai.data.datasets import RetinalDataset
from retinai.data.manifest import build_manifest, validate_manifest_images
from retinai.eval.compare import run_model_comparison
from retinai.eval.prior_correction import run_prior_correction
from retinai.eval.repeated_cv import run_repeated_cv
from retinai.eval.test_evaluator import evaluate_on_test
from retinai.hpo.optuna_runner import run_hpo
from retinai.localization.gradcam_runner import generate_gradcam_maps
from retinai.models.factory import create_model
from retinai.train.engine import run_training


def cmd_build_manifest(args):
    cfg = load_config(args.config)
    ensure_artifact_dirs(cfg)
    df = build_manifest(cfg.data.train_dir, cfg.data.class_names)
    valid_df, bad_df = validate_manifest_images(df)
    Path(cfg.data.manifest_path).parent.mkdir(parents=True, exist_ok=True)
    valid_df.to_csv(cfg.data.manifest_path, index=False)
    bad_path = str(Path(cfg.data.manifest_path).with_name("quarantine_bad_images.csv"))
    bad_df.to_csv(bad_path, index=False)
    print(f"manifest={cfg.data.manifest_path} valid={len(valid_df)} bad={len(bad_df)}")


def cmd_hpo(args):
    cfg = load_config(args.config)
    manifest = pd.read_csv(cfg.data.manifest_path)
    study = run_hpo(cfg, manifest=manifest, model_name=args.model_name)
    print(f"best_value={study.best_value}")
    print(f"best_params={study.best_params}")


def cmd_repeat_cv(args):
    cfg = load_config(args.config)
    manifest = pd.read_csv(cfg.data.manifest_path)
    params = {
        "lr": args.lr if args.lr is not None else cfg.train.lr,
        "weight_decay": args.weight_decay if args.weight_decay is not None else cfg.train.weight_decay,
        "batch_size": args.batch_size if args.batch_size is not None else cfg.train.batch_size,
    }
    summary = run_repeated_cv(cfg, manifest, model_name=args.model_name, params=params)
    summary["model"] = args.model_name
    summary["params"] = params
    out_path = Path(cfg.runtime.artifact_root) / "repeated_cv" / args.model_name / "cv_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(summary)
    print(f"Summary saved to {out_path}")


def cmd_train_final(args):
    cfg = load_config(args.config)
    ensure_artifact_dirs(cfg)
    train_manifest = pd.read_csv(cfg.data.manifest_path)
    out_dir = args.out_dir if args.out_dir else f"{cfg.runtime.artifact_root}/final/{args.model_name}"

    if args.val_split and args.val_split > 0:
        from sklearn.model_selection import train_test_split
        train_df, val_df = train_test_split(
            train_manifest, test_size=args.val_split,
            stratify=train_manifest["class_id"], random_state=cfg.runtime.seed
        )
        train_ds = RetinalDataset(train_df, image_size=cfg.data.image_size, train=True)
        valid_ds = RetinalDataset(val_df, image_size=cfg.data.image_size, train=False)
        print(f"Val split {args.val_split}: train={len(train_df)}, val={len(val_df)}")
    else:
        train_ds = RetinalDataset(train_manifest, image_size=cfg.data.image_size, train=True)
        valid_ds = None

    model = create_model(model_name=args.model_name, num_classes=len(cfg.data.class_names), pretrained=True)
    result = run_training(
        model=model,
        train_ds=train_ds,
        valid_ds=valid_ds,
        device=cfg.runtime.device,
        out_dir=out_dir,
        epochs=args.epochs if args.epochs is not None else cfg.train.epochs,
        batch_size=args.batch_size if args.batch_size is not None else cfg.train.batch_size,
        lr=args.lr if args.lr is not None else cfg.train.lr,
        weight_decay=args.weight_decay if args.weight_decay is not None else cfg.train.weight_decay,
        mixed_precision=cfg.train.mixed_precision,
        patience=cfg.train.patience,
        num_workers=cfg.runtime.num_workers,
    )
    summary = {
        "model": args.model_name,
        "best_ckpt": result["best_ckpt"],
        "training_seconds": result["training_seconds"],
        "training_minutes": round(result["training_seconds"] / 60, 1),
    }
    summary_path = Path(out_dir) / "train_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"final_model={result['best_ckpt']}")
    print(f"training_seconds={result['training_seconds']:.1f}")
    print(f"training_minutes={result['training_seconds']/60:.1f}")


def cmd_evaluate(args):
    cfg = load_config(args.config)
    test_df, bad_df = validate_manifest_images(
        build_manifest(cfg.data.test_dir, cfg.data.class_names)
    )
    if bad_df is not None and len(bad_df):
        print(f"Skipping {len(bad_df)} corrupt test images.")
    out_dir = args.out_dir if args.out_dir else f"{cfg.runtime.artifact_root}/eval/{args.model_name}"
    evaluate_on_test(
        checkpoint_path=args.checkpoint,
        model_name=args.model_name,
        test_df=test_df,
        class_names=cfg.data.class_names,
        output_dir=out_dir,
        image_size=cfg.data.image_size,
        device=cfg.runtime.device,
        n_bootstrap=args.n_bootstrap,
    )
    print(f"Results saved to {out_dir}")


def cmd_prior_correct(args):
    cfg = load_config(args.config)
    test_df, bad_df = validate_manifest_images(
        build_manifest(cfg.data.test_dir, cfg.data.class_names)
    )
    if bad_df is not None and len(bad_df):
        print(f"Skipping {len(bad_df)} corrupt test images.")
    out_dir = args.out_dir if args.out_dir else f"{cfg.runtime.artifact_root}/prior_correction/{args.model_name}"
    run_prior_correction(
        cfg=cfg,
        checkpoint_path=args.checkpoint,
        model_name=args.model_name,
        test_df=test_df,
        output_dir=out_dir,
    )


def cmd_compare_models(args):
    cfg = load_config(args.config)
    scores_dir = f"{cfg.runtime.artifact_root}/repeated_cv"
    output_dir = f"{cfg.runtime.artifact_root}/comparison"
    run_model_comparison(scores_dir=scores_dir, output_dir=output_dir)
    print(f"Comparison saved to {output_dir}")


def cmd_localize(args):
    cfg = load_config(args.config)
    image_paths = [p.strip() for p in args.images.split(",") if p.strip()]
    generate_gradcam_maps(
        checkpoint_path=args.checkpoint,
        model_name=args.model_name,
        image_paths=image_paths,
        output_dir=args.output_dir,
        num_classes=len(cfg.data.class_names),
        class_names=cfg.data.class_names,
        image_size=cfg.data.image_size,
        device=cfg.runtime.device,
    )
    print(f"Grad-CAM maps saved to {args.output_dir}")


def build_parser():
    parser = argparse.ArgumentParser(prog="retinai")
    sub = parser.add_subparsers(dest="command", required=True)

    p_manifest = sub.add_parser("build-manifest")
    p_manifest.add_argument("--config", required=True)
    p_manifest.set_defaults(func=cmd_build_manifest)

    p_hpo = sub.add_parser("hpo")
    p_hpo.add_argument("--config", required=True)
    p_hpo.add_argument("--model-name", choices=["efficientnet_b0", "deit_tiny", "resnet18"], required=True)
    p_hpo.set_defaults(func=cmd_hpo)

    p_repeat = sub.add_parser("repeat-cv")
    p_repeat.add_argument("--config", required=True)
    p_repeat.add_argument("--model-name", choices=["efficientnet_b0", "deit_tiny", "resnet18"], required=True)
    p_repeat.add_argument("--lr", type=float)
    p_repeat.add_argument("--weight-decay", type=float)
    p_repeat.add_argument("--batch-size", type=int)
    p_repeat.set_defaults(func=cmd_repeat_cv)

    p_final = sub.add_parser("train-final")
    p_final.add_argument("--config", required=True)
    p_final.add_argument("--model-name", choices=["efficientnet_b0", "deit_tiny", "resnet18"], required=True)
    p_final.add_argument("--lr", type=float)
    p_final.add_argument("--weight-decay", type=float)
    p_final.add_argument("--batch-size", type=int)
    p_final.add_argument("--epochs", type=int)
    p_final.add_argument("--val-split", type=float, help="Fraction of training data for validation/early stopping (e.g. 0.1)")
    p_final.add_argument("--out-dir", type=str, help="Override output directory")
    p_final.set_defaults(func=cmd_train_final)

    p_eval = sub.add_parser("evaluate")
    p_eval.add_argument("--config", required=True)
    p_eval.add_argument("--checkpoint", required=True)
    p_eval.add_argument("--model-name", choices=["efficientnet_b0", "deit_tiny", "resnet18"], required=True)
    p_eval.add_argument("--n-bootstrap", type=int, default=1000)
    p_eval.add_argument("--out-dir", type=str, help="Override output directory")
    p_eval.set_defaults(func=cmd_evaluate)

    p_prior = sub.add_parser("prior-correct")
    p_prior.add_argument("--config", required=True)
    p_prior.add_argument("--checkpoint", required=True)
    p_prior.add_argument("--model-name", choices=["efficientnet_b0", "deit_tiny", "resnet18"], required=True)
    p_prior.add_argument("--out-dir", type=str)
    p_prior.set_defaults(func=cmd_prior_correct)

    p_compare = sub.add_parser("compare-models")
    p_compare.add_argument("--config", required=True)
    p_compare.set_defaults(func=cmd_compare_models)

    p_localize = sub.add_parser("localize")
    p_localize.add_argument("--config", required=True)
    p_localize.add_argument("--checkpoint", required=True)
    p_localize.add_argument("--model-name", choices=["efficientnet_b0", "deit_tiny", "resnet18"], required=True)
    p_localize.add_argument("--images", required=True, help="Comma-separated image paths")
    p_localize.add_argument("--output-dir", required=True)
    p_localize.set_defaults(func=cmd_localize)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

