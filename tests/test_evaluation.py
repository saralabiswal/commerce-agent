"""
Tests for the evaluation framework — ContentScorer and HumanFeedbackStore.
"""
import pytest

# ── ContentScorer ─────────────────────────────────────────────────────────────

class TestContentScorer:

    def test_score_returns_report(self, sample_optimized_content, sample_requirements):
        from evaluation.content_scorer import ContentScorer, ContentScoreReport
        scorer = ContentScorer()
        report = scorer.score(sample_optimized_content, sample_requirements, previous_score=58.0)
        assert isinstance(report, ContentScoreReport)
        assert 0 <= report.overall_score <= 100
        assert report.grade in ("A", "B", "C", "D", "F")

    def test_score_metrics_all_present(self, sample_optimized_content, sample_requirements):
        """All 8 metric dimensions should be present in report."""
        from evaluation.content_scorer import ContentScorer
        scorer = ContentScorer()
        report = scorer.score(sample_optimized_content, sample_requirements)
        metric_names = {m.name for m in report.metrics}
        expected = {
            "title_compliance", "keyword_inclusion", "bullet_count",
            "bullet_length", "readability", "brand_safety",
            "specificity", "improvement_delta",
        }
        assert expected.issubset(metric_names)

    def test_score_title_too_long_fails(self, sample_requirements):
        from evaluation.content_scorer import ContentScorer
        scorer = ContentScorer()
        bad_content = {
            "title": "X" * 210,
            "bullet_points": ["b1", "b2", "b3", "b4", "b5"],
            "description": "desc",
            "backend_keywords": "kw",
        }
        report = scorer.score(bad_content, sample_requirements)
        title_metric = next(m for m in report.metrics if m.name == "title_compliance")
        assert not title_metric.passed

    def test_score_bullet_count_wrong_fails(self, sample_requirements):
        from evaluation.content_scorer import ContentScorer
        scorer = ContentScorer()
        bad_content = {
            "title": "SoundWave Wireless Headphones Active Noise Cancellation",
            "bullet_points": ["Only one bullet"],  # Wrong count
            "description": "Description",
            "backend_keywords": "keywords",
        }
        report = scorer.score(bad_content, sample_requirements)
        bullet_metric = next(m for m in report.metrics if m.name == "bullet_count")
        assert not bullet_metric.passed

    def test_score_keyword_inclusion_correct_keywords(self, sample_requirements):
        """Content with top keywords should get high keyword inclusion score."""
        from evaluation.content_scorer import ContentScorer
        scorer = ContentScorer()
        content = {
            "title": "Wireless Bluetooth Headphones Noise Cancelling Over Ear with Microphone",
            "bullet_points": [
                "Premium wireless headphones with bluetooth technology",
                "Noise cancelling headphones over ear design",
                "Headphones with microphone for clear calls",
                "Active noise cancellation headphones premium quality",
                "Over ear headphones comfortable fit",
            ],
            "description": "Bluetooth headphones wireless noise cancelling headphones with microphone.",
            "backend_keywords": "wireless bluetooth headphones",
        }
        report = scorer.score(content, sample_requirements)
        kw_metric = next(m for m in report.metrics if m.name == "keyword_inclusion")
        assert kw_metric.score >= 80

    def test_score_brand_safety_medical_claim_fails(self, sample_requirements):
        from evaluation.content_scorer import ContentScorer
        scorer = ContentScorer()
        content = {
            "title": "Headphones that cures fatigue",
            "bullet_points": ["These headphones treats hearing loss"] + ["bullet"] * 4,
            "description": "A medical device that cures tinnitus.",
            "backend_keywords": "headphones",
        }
        report = scorer.score(content, sample_requirements)
        safety_metric = next(m for m in report.metrics if m.name == "brand_safety")
        assert not safety_metric.passed

    def test_score_specificity_with_measurements(self, sample_requirements):
        """Content with numeric specs should score higher on specificity."""
        from evaluation.content_scorer import ContentScorer
        scorer = ContentScorer()
        specific_content = {
            "title": "SoundWave Headphones 40mm Bluetooth 5.2 30 Hour Battery",
            "bullet_points": [
                "40mm dynamic drivers deliver 20Hz-20kHz frequency response",
                "30 hours battery life, 10 minutes charges 3 hours",
                "250g lightweight with IPX4 splash resistance rating",
                "Bluetooth 5.2 connects to 2 devices simultaneously",
                "Dual beamforming microphones with echo cancellation",
            ],
            "description": "Professional 40mm driver headphones with 30-hour battery.",
            "backend_keywords": "headphones bluetooth 5.2",
        }
        report = scorer.score(specific_content, sample_requirements)
        spec_metric = next(m for m in report.metrics if m.name == "specificity")
        assert spec_metric.score >= 70

    def test_score_improvement_delta_positive(self, sample_optimized_content, sample_requirements):
        """Optimized content should score higher than original (58.0)."""
        from evaluation.content_scorer import ContentScorer
        scorer = ContentScorer()
        report = scorer.score(sample_optimized_content, sample_requirements, previous_score=58.0)
        # Optimized content should improve over the weak original
        assert report.improvement_delta > 0

    def test_score_quality_gate_threshold(self, sample_optimized_content, sample_requirements):
        """ContentScoreReport should correctly evaluate quality gate."""
        from evaluation.content_scorer import ContentScorer
        scorer = ContentScorer()
        report = scorer.score(sample_optimized_content, sample_requirements)
        # Quality gate is determined by overall_score vs threshold
        if report.overall_score >= report.quality_gate_threshold:
            assert report.passed_quality_gate is True
        else:
            assert report.passed_quality_gate is False

    def test_score_passed_failed_metrics_partition(self, sample_optimized_content, sample_requirements):
        """passed_metrics + failed_metrics should equal all metrics."""
        from evaluation.content_scorer import ContentScorer
        scorer = ContentScorer()
        report = scorer.score(sample_optimized_content, sample_requirements)
        total = len(report.metrics)
        assert len(report.passed_metrics) + len(report.failed_metrics) == total

    def test_score_grade_boundaries(self, sample_requirements):
        """Grade should map correctly from score ranges."""
        from evaluation.content_scorer import ContentScorer
        scorer = ContentScorer()

        # Test a few grade boundaries
        for score, expected_grade in [(95, "A"), (85, "B"), (75, "C"), (65, "D"), (50, "F")]:
            grade = scorer._to_grade(score)
            assert grade == expected_grade


# ── HumanFeedbackStore ────────────────────────────────────────────────────────

class TestHumanFeedbackStore:

    @pytest.mark.asyncio
    async def test_record_and_retrieve_feedback(self, temp_db):
        """Should store and retrieve feedback correctly."""
        from evaluation.human_feedback import HumanFeedbackStore
        from scripts.init_db import ensure_db

        await ensure_db(temp_db)

        store = HumanFeedbackStore(db_url=temp_db)
        feedback_id = await store.record_feedback(
            run_id="test-run-001",
            sku="DEMO-SKU-001",
            rating=1,
            comment="Great output!",
            field="overall",
        )

        assert isinstance(feedback_id, str)

        all_feedback = await store.get_feedback("test-run-001")
        assert len(all_feedback) == 1
        assert all_feedback[0]["rating"] == 1
        assert all_feedback[0]["comment"] == "Great output!"

    @pytest.mark.asyncio
    async def test_feedback_summary(self, temp_db):
        """Summary should correctly aggregate thumbs up/down."""
        from evaluation.human_feedback import HumanFeedbackStore
        from scripts.init_db import ensure_db

        await ensure_db(temp_db)
        store = HumanFeedbackStore(db_url=temp_db)

        # 3 thumbs up, 1 thumbs down
        for i in range(3):
            await store.record_feedback(f"run-{i}", "SKU-001", 1)
        await store.record_feedback("run-3", "SKU-001", -1)

        summary = await store.get_summary()
        assert summary["total_feedback"] == 4
        assert summary["thumbs_up"] == 3
        assert summary["thumbs_down"] == 1
        assert summary["approval_rate"] == 75.0

    @pytest.mark.asyncio
    async def test_feedback_empty_summary(self, temp_db):
        """Empty database should return zero counts without error."""
        from evaluation.human_feedback import HumanFeedbackStore
        from scripts.init_db import ensure_db

        await ensure_db(temp_db)
        store = HumanFeedbackStore(db_url=temp_db)

        summary = await store.get_summary()
        assert summary["total_feedback"] == 0
        assert summary["approval_rate"] == 0.0
