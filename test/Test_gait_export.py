"""Tests for automatic threshold-based GA training-data export."""

import io
import tempfile
import unittest
from collections.abc import Callable
from contextlib import chdir, redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np

from GA import GA, try_all_ga_configs


class TestGaitExport(unittest.TestCase):
    """Check fitness filtering, run grouping, and the shared data format."""

    def test_save_training_gaits(self) -> None:
        """Keep strictly qualifying individuals across generations, once each."""
        generations = np.zeros((2, 3, 300, 8, 3))
        generations[0, 1, :, 0, 0] = 0.1
        generations[0, 2, :, 0, 0] = 0.2
        generations[1, 0] = generations[0, 2]  # Surviving elite, not new training data.
        generations[1, 1, :, 0, 0] = 0.3
        generations[1, 2, :, 0, 0] = 0.4
        fitness = np.array([[0.5, 0.4, 0.7], [0.7, 0.8, -1]])
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "data"
            path = GA.save_training_gaits(generations, fitness, output, threshold=0.5)
            assert path is not None
            with np.load(path, allow_pickle=False) as archive:
                np.testing.assert_array_equal(archive["gaits"], generations[[0, 1], [2, 1]])
                np.testing.assert_array_equal(archive["fitness"], [0.7, 0.8])
                np.testing.assert_array_equal(archive["generation_indices"], [0, 1])
            # No empty archive should be written when no individual qualifies.
            self.assertIsNone(GA.save_training_gaits(generations, fitness, output, threshold=1))
            self.assertEqual(len(list(output.glob("*.npz"))), 1)
            # Separate runs must get separate archives, even with identical results.
            GA.save_training_gaits(generations, fitness, output, threshold=0.5)
            self.assertEqual(len(list(output.glob("*.npz"))), 2)

    def test_run_GA_has_no_export_or_logging_side_effects(self) -> None:
        """Running evolution alone neither exports data nor prints application logs."""
        messages = io.StringIO()
        with (
            redirect_stdout(messages),
            patch.object(
                GA, "save_training_gaits", side_effect=AssertionError("unexpected export")
            ),
        ):
            history = GA.run_GA(3, 1, 1, 1, 0.001, GA.tournament_selection, 2, initial_std=0.001)
        self.assertEqual(history.shape, (2, 3, 300, 8, 3))
        self.assertEqual(messages.getvalue(), "")

    def test_run_GA_generation_callback(self) -> None:
        """Report every scored generation, including initial-only and final cases."""
        scores = []
        generation_numbers = []

        def collect_generation(generation: int, fitness: np.ndarray) -> None:
            """Collect scores without exporting data or changing evolution."""
            generation_numbers.append(generation)
            scores.append(fitness.copy())

        for count in (0, 1):
            with self.subTest(generations=count):
                scores.clear()
                generation_numbers.clear()
                history = GA.run_GA(
                    3,
                    count,
                    1,
                    1,
                    0.001,
                    GA.tournament_selection,
                    2,
                    initial_std=0.001,
                    on_generation=collect_generation,
                )
                self.assertEqual(generation_numbers, list(range(count + 1)))
                for population, fitness in zip(history, scores):
                    np.testing.assert_array_equal(fitness, GA.evaluate_fitness(population))

    def test_main_exports_and_reports(self) -> None:
        """Main owns export settings, progress messages, and export success/failure."""
        run_GA = GA.run_GA
        histories = []

        def small_run(
            *, on_generation: Callable[[int, np.ndarray], None], **kwargs: object
        ) -> np.ndarray:
            """Exercise the real GA while keeping the application's run small."""
            history = run_GA(
                3,
                1,
                1,
                1,
                0.001,
                GA.tournament_selection,
                2,
                initial_std=0.001,
                on_generation=on_generation,
            )
            histories.append(history)
            return history

        for threshold in (-0.5, GA.MAX_FITNESS):
            with (
                self.subTest(threshold=threshold),
                tempfile.TemporaryDirectory() as directory,
                chdir(directory),
                patch.object(GA, "FITNESS_SAVE_THRESHOLD", threshold),
            ):
                output = Path(directory) / "data"
                histories.clear()

                messages = io.StringIO()
                with (
                    redirect_stdout(messages),
                    patch.object(GA, "run_GA", side_effect=small_run),
                    patch.object(GA, "GAIT_DATA_DIRECTORY", output),
                    patch("matplotlib.pyplot.show"),
                ):
                    self.assertEqual(GA.main(printy=True), 0)
                log = messages.getvalue()
                self.assertIn(str(output), log)
                self.assertIn(f"fitness > {threshold}", log)
                self.assertIn("Generation 0/", log)
                self.assertIn("Generation 1/", log)
                self.assertTrue(Path("recent_run.pkl").is_file())
                paths = list(output.glob("*.npz"))
                if threshold < GA.MAX_FITNESS:
                    self.assertEqual(len(paths), 1)
                    history = histories[0]
                    expected = np.concatenate((history[0], history[1, 1:]))
                    with np.load(paths[0], allow_pickle=False) as archive:
                        np.testing.assert_array_equal(archive["gaits"], expected)
                        np.testing.assert_array_equal(
                            archive["generation_indices"], [0, 0, 0, 1, 1]
                        )
                    self.assertIn("Saved qualifying NN training gaits", log)
                else:
                    self.assertFalse(paths)
                    self.assertIn("qualifying=0", log)
                    self.assertIn("No gaits exceeded", log)

    def test_configuration_driver_exports_runs(self) -> None:
        """The experiment caller explicitly saves data after running evolution."""
        run_GA = GA.run_GA

        def small_run(**kwargs: object) -> np.ndarray:
            """Return a small real run in place of the expensive experiment settings."""
            return run_GA(3, 0, 0, 0, 0, GA.tournament_selection, initial_std=0.001)

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with (
                patch.multiple(
                    try_all_ga_configs,
                    CO_RATES=[0],
                    MUT_RATES=[0],
                    MUT_STDS=[0],
                    SELECTION_METHODS=[GA.tournament_selection],
                ),
                patch.object(try_all_ga_configs, "run_GA", side_effect=small_run),
                patch.object(GA, "GAIT_DATA_DIRECTORY", output),
            ):
                with patch.object(GA, "FITNESS_SAVE_THRESHOLD", -np.inf):
                    results = try_all_ga_configs.try_all_permutations(42)
            self.assertEqual(len(results), 1)
            self.assertEqual(len(list(output.glob("*.npz"))), 1)


if __name__ == "__main__":
    unittest.main()
