import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from deploy.create_dustin_venv import plan_commands
from src.deployment import (
    CONSTRAINT_DISTS,
    EXPECTED_DIST_VERSIONS,
    MODEL_DIR_ENV,
    NO_DEPS_DISTS,
    STANDARD_WEIGHT_BYTES,
    SafetyError,
    TORCH_CU128_WIN_CP312_BYTES,
    assert_config_writable,
    assert_not_readonly_model_dir,
    assert_safe_to_train,
    build_resolved,
    persistent_workers,
    prepare_training_run,
    protected_reason,
    standard_weight_bytes_total,
)
from src.config import TrainConfig


def _env(root, profile, confirm=False):
    env = {"TTS_ROOT": str(root), "TTS_PROFILE": profile}
    if confirm:
        env["TTS_CONFIRM_TRAIN"] = "1"
    return env


class DeploymentTests(unittest.TestCase):
    def test_default_profile_is_isolated_dustin(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / "ft" / "repo"
            resolved = build_resolved(environ=_env(root, "dustin"), repo_root=repo)
            self.assertEqual(resolved.profile, "dustin")
            self.assertEqual(resolved.blocked_reasons, [])
            self.assertFalse(resolved.config.is_turbo)
            self.assertFalse(resolved.config.is_nano)
            self.assertEqual(resolved.config.batch_size, 1)
            self.assertEqual(resolved.config.lora_r, 8)
            self.assertEqual(resolved.config.dataloader_num_workers, 0)
            self.assertFalse(persistent_workers(resolved.config))
            self.assertTrue(resolved.config.output_dir.endswith(os.path.join("dustin", "output")))
            self.assertTrue(resolved.config.model_dir.endswith(os.path.join("ft", "repo", "pretrained_models")))
            self.assertNotIn("pretrained_models_turbo", resolved.config.model_dir)
            self.assertIsNone(protected_reason(resolved.config.output_dir))
            self.assertIsNone(protected_reason(resolved.config.model_dir))

    def test_upstream_keeps_committed_nano_defaults(self):
        fresh = TrainConfig()
        resolved = build_resolved(environ={"TTS_PROFILE": "upstream"})
        cfg = resolved.config
        self.assertEqual(cfg.is_turbo, fresh.is_turbo)
        self.assertEqual(cfg.is_nano, fresh.is_nano)
        self.assertEqual(cfg.lora_r, fresh.lora_r)
        self.assertEqual(cfg.batch_size, fresh.batch_size)
        self.assertEqual(cfg.num_epochs, fresh.num_epochs)
        self.assertEqual(cfg.logging_steps, fresh.logging_steps)
        self.assertEqual(cfg.dataloader_num_workers, fresh.dataloader_num_workers)
        self.assertTrue(persistent_workers(cfg))
        self.assertTrue(cfg.model_dir.endswith("pretrained_models_turbo"))
        self.assertTrue(cfg.output_dir.endswith("chatterbox_output_turbo"))
        self.assertTrue(cfg.json_format)
        self.assertFalse(cfg.ljspeech)
        self.assertEqual(resolved.blocked_reasons, [])

    def test_lewis_profile_is_preserved_and_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "ft" / "data" / "wavs").mkdir(parents=True)
            (root / "ft" / "output").mkdir()
            csv_path = root / "ft" / "data" / "metadata.csv"
            csv_path.write_text("a.wav|hello|hello\n", encoding="utf-8")
            resolved = build_resolved(environ=_env(root, "lewis", confirm=True), repo_root=root / "ft" / "repo")
            self.assertFalse(resolved.config.is_turbo)
            self.assertEqual(resolved.config.lora_r, 32)
            self.assertEqual(resolved.config.lora_alpha, 64)
            self.assertEqual(resolved.config.batch_size, 8)
            self.assertEqual(resolved.config.grad_accum, 2)
            self.assertEqual(resolved.config.num_epochs, 20)
            self.assertEqual(resolved.config.save_steps, 100)
            self.assertEqual(resolved.config.logging_steps, 5)
            self.assertEqual(resolved.config.save_total_limit, 4)
            self.assertEqual(resolved.config.dataloader_num_workers, 0)
            self.assertTrue(resolved.blocked_reasons)
            with self.assertRaises(SafetyError):
                prepare_training_run(environ=_env(root, "lewis", confirm=True), repo_root=root / "ft" / "repo")

    def test_dustin_confirm_and_dataset_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / "repo"
            with self.assertRaises(SafetyError) as denied:
                prepare_training_run(environ=_env(root, "dustin"), repo_root=repo)
            self.assertIn("TTS_CONFIRM_TRAIN", str(denied.exception))
            data = root / "dustin" / "data"
            (data / "wavs").mkdir(parents=True)
            (data / "metadata.csv").write_text("clip.wav|hello|hello\n", encoding="utf-8")
            resolved = prepare_training_run(environ=_env(root, "dustin", confirm=True), repo_root=repo)
            self.assertTrue(resolved.config.csv_path.endswith("metadata.csv"))
            self.assertFalse((root / "ft").exists())

    def test_protected_names_and_no_mkdir(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cases = [
                root / "voice_final" / "ref.wav",
                root / "sets" / "demon" / "wavs",
                root / "sets" / "toadpipe" / "metadata.csv",
                root / "ft" / "logs" / "train.log",
                root / "FT" / "Data" / "wavs",
            ]
            for path in cases:
                self.assertIsNotNone(protected_reason(path), path)
            config = TrainConfig()
            target = root / "ft" / "output"
            config.output_dir = str(target)
            config.preprocessed_dir = str(root / "ft" / "data" / "preprocess")
            config.wav_dir = str(root / "ft" / "data" / "wavs")
            config.csv_path = str(root / "ft" / "data" / "metadata.csv")
            with self.assertRaises(SafetyError):
                assert_config_writable(config)
            self.assertFalse(target.exists())

    def test_install_plan_keeps_cu128_and_skips_torch_2_6(self):
        commands = plan_commands(sys.executable)
        flat = "\n".join(" ".join(command) for command in commands)
        self.assertIn("torch==2.11.0+cu128", flat)
        self.assertIn("https://download.pytorch.org/whl/cu128", flat)
        self.assertIn("--no-deps", flat)
        self.assertIn("chatterbox-tts==0.1.7", flat)
        self.assertIn("setuptools==81.0.0", flat)
        self.assertNotIn("torch==2.6.0", flat)
        self.assertNotIn("setuptools==84", flat)
        self.assertNotIn("base-freeze", flat)
        self.assertNotIn("venv-cbft", flat)
        self.assertEqual(TORCH_CU128_WIN_CP312_BYTES, 2_753_189_216)
        self.assertEqual(standard_weight_bytes_total(), 3_192_011_511)

    def test_requirement_pins_match_manifest(self):
        requirements = (REPO / "deploy" / "requirements-dustin.txt").read_text(encoding="utf-8")
        constraints = (REPO / "deploy" / "constraints-dustin.txt").read_text(encoding="utf-8")

        def normalize(name):
            return name.lower().replace("_", "-")

        pinned = {}
        for line in requirements.splitlines():
            line = line.split("#", 1)[0].strip()
            if "==" in line:
                name, version = line.split("==", 1)
                pinned[normalize(name)] = version
        constrained = {}
        for line in constraints.splitlines():
            line = line.split("#", 1)[0].strip()
            if "==" in line:
                name, version = line.split("==", 1)
                constrained[normalize(name)] = version
        for name, version in EXPECTED_DIST_VERSIONS.items():
            key = normalize(name)
            if key in NO_DEPS_DISTS or key in CONSTRAINT_DISTS:
                self.assertNotIn(key, pinned)
            else:
                self.assertEqual(pinned[key], version)
        for name in CONSTRAINT_DISTS:
            self.assertEqual(constrained[normalize(name)], EXPECTED_DIST_VERSIONS[name])
        self.assertNotIn("chatterbox-tts", pinned)

    def test_repository_has_no_personal_paths(self):
        banned = (
            "fl" + "ame",
            "Gro" + "k W" + "ork",
            "C:\\" + "tts",
            "C:/" + "tts",
            "ref_" + "studio",
            "phi" + "lia",
        )
        offenders = []
        for path in REPO.rglob("*"):
            if not path.is_file() or ".git" in path.parts or "__pycache__" in path.parts:
                continue
            if path.suffix.lower() in {".wav", ".mp3", ".pt", ".safetensors", ".pyc"}:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            for token in banned:
                if token in text:
                    offenders.append(f"{path.relative_to(REPO)} contains a private token")
        self.assertEqual(offenders, [])

    def test_scripts_do_not_install_or_train_on_help(self):
        dry = subprocess.run(
            [sys.executable, str(REPO / "deploy" / "create_dustin_venv.py")],
            cwd=REPO,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(dry.returncode, 0, dry.stderr)
        self.assertIn("No packages were installed", dry.stdout)
        help_render = subprocess.run(
            [sys.executable, str(REPO / "render_ft.py"), "--help"],
            cwd=REPO,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(help_render.returncode, 0, help_render.stderr)
        blocked = subprocess.run(
            [sys.executable, str(REPO / "render_book.py"), "--text", "a", "--out", "b", "--ref", "c", "--checkpoint", "d"],
            cwd=REPO,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(blocked.returncode, 2)
        self.assertIn("whisper", blocked.stderr.lower())
        setup = subprocess.run(
            [sys.executable, str(REPO / "setup.py")],
            cwd=REPO,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(setup.returncode, 2, setup.stderr)
        self.assertIn("Refusing to download", setup.stdout)
        smoke = subprocess.run(
            [sys.executable, str(REPO / "deploy" / "smoke_check.py"), "--paths-only"],
            cwd=REPO,
            capture_output=True,
            text=True,
            check=False,
            env={**os.environ, "TTS_PROFILE": "dustin", "TTS_ROOT": ""},
        )
        self.assertEqual(smoke.returncode, 0, smoke.stdout + smoke.stderr)
        self.assertIn("profile dustin:", smoke.stdout)
        self.assertIn("did not train", smoke.stdout)

    def test_train_patches_precede_chatterbox_import(self):
        text = (REPO / "train.py").read_text(encoding="utf-8")
        self.assertLess(text.index("apply_training_patches()"), text.index("src.chatterbox_"))

    def _write_standard_weights(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        for name in STANDARD_WEIGHT_BYTES:
            (directory / name).write_bytes(b"weight")

    def test_model_dir_override_reuses_sibling_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            code = root / "ft" / "repo-dustin"
            weights = root / "ft" / "repo" / "pretrained_models"
            self._write_standard_weights(weights)
            data = root / "dustin" / "data"
            (data / "wavs").mkdir(parents=True)
            (data / "metadata.csv").write_text("clip.wav|hello|hello\n", encoding="utf-8")
            before = {path.name: path.read_bytes() for path in weights.iterdir()}
            env = {
                "TTS_ROOT": str(root),
                "TTS_PROFILE": "dustin",
                "TTS_CONFIRM_TRAIN": "1",
                MODEL_DIR_ENV: str(weights),
            }
            resolved = prepare_training_run(environ=env, repo_root=code)
            self.assertTrue(resolved.model_dir_override)
            self.assertEqual(Path(resolved.config.model_dir), weights.resolve())
            self.assertEqual(resolved.blocked_reasons, [])
            self.assertFalse((code / "pretrained_models").exists())
            self.assertEqual(before, {path.name: path.read_bytes() for path in weights.iterdir()})
            self.assertTrue(resolved.config.output_dir.endswith(os.path.join("dustin", "output")))
            resolved.config.output_dir = str(weights / "new_output")
            with self.assertRaises(SafetyError):
                assert_config_writable(resolved.config, environ=env)
            self.assertFalse((weights / "new_output").exists())
            self.assertEqual((weights / "ve.safetensors").read_bytes(), b"weight")

    def test_model_dir_override_rejects_protected_symlink_and_legacy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            code = root / "ft" / "repo-dustin"
            weights = root / "ft" / "repo" / "pretrained_models"
            self._write_standard_weights(weights)
            link = root / "weights-link"
            link.symlink_to(weights, target_is_directory=True)
            rejected = [
                root / "ft" / "data" / "pretrained_models",
                root / "ft" / "output",
                root / "ft" / "logs",
                root / "voice_final",
                root / "demon" / "pretrained_models",
                root / "toadpipe",
                link,
                Path("C:" + "\\" + "tts") / "pretrained_models",
            ]
            for candidate in rejected:
                env = {"TTS_ROOT": str(root), "TTS_PROFILE": "dustin", MODEL_DIR_ENV: str(candidate)}
                with self.assertRaises(SafetyError, msg=str(candidate)):
                    build_resolved(environ=env, repo_root=code)
            self.assertFalse((code / "pretrained_models").exists())
            self.assertFalse(any(path.is_symlink() for path in code.rglob("*")))
            incomplete = root / "ft" / "repo" / "incomplete"
            incomplete.mkdir()
            env = {"TTS_ROOT": str(root), "TTS_PROFILE": "dustin", MODEL_DIR_ENV: str(incomplete)}
            with self.assertRaises(SafetyError):
                build_resolved(environ=env, repo_root=code)
            self.assertEqual(list(incomplete.iterdir()), [])
            env = {"TTS_ROOT": str(root), "TTS_PROFILE": "dustin", MODEL_DIR_ENV: str(weights)}
            with self.assertRaises(SafetyError):
                assert_not_readonly_model_dir(weights / "ve.safetensors", environ=env)
            with self.assertRaises(SafetyError):
                assert_not_readonly_model_dir(weights, environ=env)
            elsewhere = code / "pretrained_models_turbo"
            assert_not_readonly_model_dir(elsewhere, environ=env)
            self.assertFalse(elsewhere.exists())
            self.assertEqual((weights / "ve.safetensors").read_bytes(), b"weight")
            setup_env = os.environ.copy()
            setup_env["TTS_ALLOW_SETUP"] = "1"
            setup_env[MODEL_DIR_ENV] = str(weights)
            setup = subprocess.run(
                [sys.executable, str(REPO / "setup.py")],
                cwd=weights,
                env=setup_env,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(setup.returncode, 2, setup.stderr)
            self.assertIn("read-only", setup.stdout)
            self.assertEqual((weights / "ve.safetensors").read_bytes(), b"weight")
            self.assertFalse((weights / "pretrained_models_turbo").exists())

    def test_upstream_training_gate_does_not_require_dustin_confirm(self):
        resolved = build_resolved(environ={"TTS_PROFILE": "upstream"})
        assert_safe_to_train(resolved, environ={"TTS_PROFILE": "upstream"})


if __name__ == "__main__":
    unittest.main()
