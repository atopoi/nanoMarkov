# MM Framework Makefile - Reorganized Structure
# Run from project root directory

SCALES = 10 100 1000
SPARSE_SCALES = 100-sparse-75 1000-sparse-75
ALL_MM_SCALES = $(SCALES) $(SPARSE_SCALES)
MM_EXTRA_BASES = 100 100-sparse-75
SEEDS = 42
#SEEDS = 42 123 999
EXTRA_VARIANTS = c1 c2 c3 c4 c5 c6 c7 c8 c9 c10 c11 c12

# Top-level targets
all: MM-10 MM-100

# Scale completion targets
MM-10: trainings/MM/MM-10/MM-10.done
MM-100: trainings/MM/MM-100/MM-100.done  
MM-1000: trainings/MM/MM-1000/MM-1000.done
MM-100-sparse-75: trainings/MM/MM-100-sparse-75/MM-100-sparse-75.done
MM-1000-sparse-75: trainings/MM/MM-1000-sparse-75/MM-1000-sparse-75.done

TRAIN = train_mm.py

# Function to generate seed targets for a scale
define make-seed-targets
trainings/MM/MM-$(1)/MM-$(1)-$(2)/model.done: data/mm$(1)/train.bin
	@echo "🚀 Training MM-$(1) seed $(2)"
	@mkdir -p $$(dir $$@)
	@echo "Start: $$$$(date)" > $$(dir $$@)training.start
	@python configs/mm_config.py $(1) $(2) > $$(dir $$@)config.py
	@echo "Config: mm$(1)-s$(2)" >> $$(dir $$@)training.start
	@echo "Status: TRAINING" >> $$(dir $$@)training.start
	@python -u $(TRAIN) $$(dir $$@)config.py > $$(dir $$@)training.log 2>&1
	@echo "Completed: $$$$(date)" > $$@
	@echo "Status: SUCCESS" >> $$@
	@grep -o "step [0-9]*.*loss [0-9.]*" $$(dir $$@)training.log | tail -1 >> $$@ || echo "Final loss: unknown" >> $$@
	@echo "✅ Training completed for MM-$(1) seed $(2)"
endef

# Generate all seed targets for each scale
$(foreach scale,$(SCALES),$(foreach seed,$(SEEDS),$(eval $(call make-seed-targets,$(scale),$(seed)))))

# Function to generate sparse seed targets
define make-sparse-seed-targets
trainings/MM/MM-$(1)/MM-$(1)-$(2)/model.done: data/mm$(1)/train.bin
	@echo "🚀 Training MM-$(1) seed $(2) (SPARSE)"
	@mkdir -p $$(dir $$@)
	@echo "Start: $$$$(date)" > $$(dir $$@)training.start
	@python configs/mm_config.py $(1) $(2) > $$(dir $$@)config.py
	@echo "Config: mm$(1)-s$(2) (SPARSE)" >> $$(dir $$@)training.start
	@echo "Status: TRAINING" >> $$(dir $$@)training.start
	@python -u $(TRAIN) $$(dir $$@)config.py > $$(dir $$@)training.log 2>&1
	@echo "Completed: $$$$(date)" > $$@
	@echo "Status: SUCCESS" >> $$@
	@grep -o "step [0-9]*.*loss [0-9.]*" $$(dir $$@)training.log | tail -1 >> $$@ || echo "Final loss: unknown" >> $$@
	@echo "✅ Training completed for MM-$(1) seed $(2) (SPARSE)"
endef

# Generate all sparse seed targets
$(foreach scale,$(SPARSE_SCALES),$(foreach seed,$(SEEDS),$(eval $(call make-sparse-seed-targets,$(scale),$(seed)))))

# Scale completion targets (explicit dependencies work better than functions in pattern rules)
trainings/MM/MM-10/MM-10.done: trainings/MM/MM-10/MM-10-42/model.done
	@echo "🎯 Completing scale MM-10 with seeds: $(SEEDS)"
	@mkdir -p $(dir $@)
	@echo "Scale MM-10 completed: $(SEEDS)" > $@
	@echo "Completed: $$(date)" >> $@
	@echo "✅ Scale MM-10 fully complete"

trainings/MM/MM-100/MM-100.done: trainings/MM/MM-100/MM-100-42/model.done
	@echo "🎯 Completing scale MM-100 with seeds: $(SEEDS)"
	@mkdir -p $(dir $@)
	@echo "Scale MM-100 completed: $(SEEDS)" > $@
	@echo "Completed: $$(date)" >> $@
	@echo "✅ Scale MM-100 fully complete"

trainings/MM/MM-1000/MM-1000.done: trainings/MM/MM-1000/MM-1000-42/model.done
	@echo "🎯 Completing scale MM-1000 with seeds: $(SEEDS)"
	@mkdir -p $(dir $@)
	@echo "Scale MM-1000 completed: $(SEEDS)" > $@
	@echo "Completed: $$(date)" >> $@
	@echo "✅ Scale MM-1000 fully complete"

trainings/MM/MM-100-sparse-75/MM-100-sparse-75.done: trainings/MM/MM-100-sparse-75/MM-100-sparse-75-42/model.done
	@echo "🎯 Completing scale MM-100-sparse-75 with seeds: $(SEEDS)"
	@mkdir -p $(dir $@)
	@echo "Scale MM-100-sparse-75 completed: $(SEEDS)" > $@
	@echo "Completed: $$(date)" >> $@
	@echo "✅ Scale MM-100-sparse-75 fully complete"

trainings/MM/MM-1000-sparse-75/MM-1000-sparse-75.done: trainings/MM/MM-1000-sparse-75/MM-1000-sparse-75-42/model.done
	@echo "🎯 Completing scale MM-1000-sparse-75 with seeds: $(SEEDS)"
	@mkdir -p $(dir $@)
	@echo "Scale MM-1000-sparse-75 completed: $(SEEDS)" > $@
	@echo "Completed: $$(date)" >> $@
	@echo "✅ Scale MM-1000-sparse-75 fully complete"

# Function to generate evaluation targets
define make-eval-targets
trainings/MM/MM-$(1)/MM-$(1)-$(2)/model_eval.txt: trainings/MM/MM-$(1)/MM-$(1)-$(2)/model.done
	@echo "📊 Evaluating MM-$(1) seed $(2)"
	@python scripts/mm_eval.py $(1) --seeds $(2) > $$@
	@echo "✅ Evaluation completed for MM-$(1) seed $(2)"
endef

# Generate all evaluation targets (regular + sparse)
$(foreach scale,$(ALL_MM_SCALES),$(foreach seed,$(SEEDS),$(eval $(call make-eval-targets,$(scale),$(seed)))))

# Function to generate Metrics 4 & 5 targets for MM scales
define make-metrics45-targets
trainings/MM/MM-$(1)/MM-$(1)-$(2)/mm_metrics_4_5.json: trainings/MM/MM-$(1)/MM-$(1)-$(2)/model.done
	@echo "🔬 Running Metrics 4 & 5 for MM-$(1) seed $(2)"
	@if echo "$(1)" | grep -q "sparse"; then \
		python scripts/mm_eval_transition_matrix.py \
			--transformer_path trainings/MM/MM-$(1)/MM-$(1)-$(2)/ckpt.pt \
			--formal_model_path data/mm$(1)/model.pkl \
			--output $$@; \
	else \
		python scripts/mm_eval_transition_matrix.py \
			--transformer_path trainings/MM/MM-$(1)/MM-$(1)-$(2)/ckpt.pt \
			--formal_model_path data/mm$(1)/mm$(1)-model.pkl \
			--output $$@; \
	fi
	@echo "✅ Metrics 4 & 5 completed for MM-$(1) seed $(2)"
endef

# Generate all Metrics 4 & 5 targets (regular + sparse)
$(foreach scale,$(ALL_MM_SCALES),$(foreach seed,$(SEEDS),$(eval $(call make-metrics45-targets,$(scale),$(seed)))))

# Function to generate MM-extra targets for any base MM model
define make-mm-extra-targets
# $(1) = base_id (e.g., "100", "100-sparse-75", "10")
# $(2) = variant (e.g., "c12")
# $(3) = seed (e.g., "42")

trainings/MM/MM-$(1)-extra/MM-$(1)-extra-$(2)-s$(3)/training.done: data/mm$(1)/train.bin
	@echo "🔬 Training MM-$(1)-extra $(2) variant (seed $(3))"
	@mkdir -p $$(dir $$@)
	@if echo "$(1)" | grep -q "sparse"; then \
		suffix=$$$$(echo "-$(1)" | sed 's/^-[0-9]*//'); \
		python configs/mm_config_extra.py $(2) $(3) $$$$suffix > $$(dir $$@)config.py; \
	else \
		python configs/mm_config_extra.py $(2) $(3) > $$(dir $$@)config.py; \
	fi
	@python -u $(TRAIN) $$(dir $$@)config.py > $$(dir $$@)training.log 2>&1
	@echo "Completed: $$$$(date)" > $$@
	@echo "Status: SUCCESS" >> $$@
	@grep -o "step [0-9]*.*loss [0-9.]*" $$(dir $$@)training.log | tail -1 >> $$@ || echo "Final loss: unknown" >> $$@
	@echo "✅ MM-$(1)-extra $(2) completed"
endef

# Generate all MM-extra targets for regular and sparse bases
$(foreach base,$(MM_EXTRA_BASES),$(foreach variant,$(EXTRA_VARIANTS),$(foreach seed,$(SEEDS),$(eval $(call make-mm-extra-targets,$(base),$(variant),$(seed))))))

# Function to generate MM-extra evaluation targets
define make-mm-extra-eval-targets
# $(1) = base_id (e.g., "100", "100-sparse-75")
# $(2) = variant (e.g., "c12") 
# $(3) = seed (e.g., "42")

trainings/MM/MM-$(1)-extra/MM-$(1)-extra-$(2)-s$(3)/model_eval.txt: trainings/MM/MM-$(1)-extra/MM-$(1)-extra-$(2)-s$(3)/training.done
	@echo "📊 Evaluating MM-$(1)-extra $(2) variant (seed $(3))"
	@python scripts/mm_eval.py $(2) > $$@
	@echo "✅ Evaluation completed for MM-$(1)-extra $(2)"
endef

# Function to generate MM-extra Metrics 4 & 5 targets
define make-mm-extra-metrics45-targets
# $(1) = base_id (e.g., "100", "100-sparse-75")
# $(2) = variant (e.g., "c12")
# $(3) = seed (e.g., "42")

trainings/MM/MM-$(1)-extra/MM-$(1)-extra-$(2)-s$(3)/mm_metrics_4_5.json: trainings/MM/MM-$(1)-extra/MM-$(1)-extra-$(2)-s$(3)/training.done
	@echo "🔬 Running Metrics 4 & 5 for MM-$(1)-extra $(2) variant (seed $(3))"
	@if echo "$(1)" | grep -q "sparse"; then \
		python scripts/mm_eval_transition_matrix.py \
			--transformer_path trainings/MM/MM-$(1)-extra/MM-$(1)-extra-$(2)-s$(3)/ckpt.pt \
			--formal_model_path data/mm$(1)/model.pkl \
			--output $$@; \
	else \
		python scripts/mm_eval_transition_matrix.py \
			--transformer_path trainings/MM/MM-$(1)-extra/MM-$(1)-extra-$(2)-s$(3)/ckpt.pt \
			--formal_model_path data/mm$(1)/mm$(1)-model.pkl \
			--output $$@; \
	fi
	@echo "✅ Metrics 4 & 5 completed for MM-$(1)-extra $(2)"
endef

# Generate all MM-extra evaluation and metrics targets
$(foreach base,$(MM_EXTRA_BASES),$(foreach variant,$(EXTRA_VARIANTS),$(foreach seed,$(SEEDS),$(eval $(call make-mm-extra-eval-targets,$(base),$(variant),$(seed))))))
$(foreach base,$(MM_EXTRA_BASES),$(foreach variant,$(EXTRA_VARIANTS),$(foreach seed,$(SEEDS),$(eval $(call make-mm-extra-metrics45-targets,$(base),$(variant),$(seed))))))

# Scale evaluation targets (explicit dependencies)
trainings/MM/MM-10/eval_summary.txt: trainings/MM/MM-10/MM-10-42/model_eval.txt
	@echo "📈 Generating evaluation summary for MM-10"
	@python scripts/mm_eval.py 10 --seeds $(SEEDS) --summary > $@
	@echo "✅ Evaluation summary completed for MM-10"

trainings/MM/MM-100/eval_summary.txt: trainings/MM/MM-100/MM-100-42/model_eval.txt
	@echo "📈 Generating evaluation summary for MM-100"
	@python scripts/mm_eval.py 100 --seeds $(SEEDS) --summary > $@
	@echo "✅ Evaluation summary completed for MM-100"

trainings/MM/MM-1000/eval_summary.txt: trainings/MM/MM-1000/MM-1000-42/model_eval.txt
	@echo "📈 Generating evaluation summary for MM-1000"
	@python scripts/mm_eval.py 1000 --seeds $(SEEDS) --summary > $@
	@echo "✅ Evaluation summary completed for MM-1000"

# Sparse MM evaluation summary targets
trainings/MM/MM-100-sparse-75/eval_summary.txt: trainings/MM/MM-100-sparse-75/MM-100-sparse-75-42/model_eval.txt
	@echo "📈 Generating evaluation summary for MM-100-sparse-75"
	@python scripts/mm_eval.py 100-sparse-75 --seeds $(SEEDS) --summary > $@
	@echo "✅ Evaluation summary completed for MM-100-sparse-75"

# Data generation with appropriate sizes
data/mm10/train.bin:
	@echo "📊 Generating MM-10 data..."
	@python scripts/mm_generate.py 10 --train 10000 --val 1000

data/mm100/train.bin:
	@echo "📊 Generating MM-100 data..."
	@python scripts/mm_generate.py 100 --train 50000 --val 5000

data/mm1000/train.bin:
	@echo "📊 Generating MM-1000 data..."
	@python scripts/mm_generate.py 1000 --train 50000 --val 5000

# Sparse MM data generation
data/mm100-sparse-75/train.bin:
	@echo "📊 Generating MM-100 SPARSE-75% data..."
	@python scripts/mm_generate.py 100 --train 50000 --val 5000 --sparsity 75

# Sparse MM data generation
data/mm1000-sparse-75/train.bin:
	@echo "📊 Generating MM-1000 SPARSE-75% data..."
	@python scripts/mm_generate.py 1000 --train 50000 --val 5000 --sparsity 75

# Parallel execution: Use make -j4 MM-100-sparse-75-FULL instead

# Status checking
status:
	@echo "📊 MM Framework Status:"
	@echo "======================"
	@for scale in $(SCALES); do \
		echo ""; \
		echo "MM-$$scale:"; \
		completed=$$(find trainings/MM/MM-$$scale/ -name "model.done" 2>/dev/null | wc -l | tr -d ' '); \
		total=3; \
		if [ $$completed -eq $$total ]; then \
			echo "  ✅ Complete ($$completed/$$total models)"; \
		elif [ $$completed -gt 0 ]; then \
			echo "  🔄 In Progress ($$completed/$$total models)"; \
			find trainings/MM/MM-$$scale/ -name "training.start" -not -path "*/model.done" 2>/dev/null | while read file; do \
				if [ -f "$$file" ]; then \
					seed=$$(basename $$(dirname $$file) | cut -d'-' -f3); \
					echo "    - Seed $$seed: Training"; \
				fi; \
			done; \
		else \
			echo "  ❌ Not started (0/$$total models)"; \
		fi; \
	done

# Clean targets
clean-MM-%:
	@echo "🧹 Cleaning MM-$* results..."
	@rm -rf trainings/MM/MM-$*
	@echo "✅ Cleaned MM-$*"

clean-all:
	@echo "🧹 Cleaning all MM results..."
	@rm -rf trainings/MM/
	@echo "✅ Cleaned all MM results"

# MM-extra evaluation targets
trainings/MM/MM-extra/%-s42/model_eval.txt: trainings/MM/MM-extra/%-s42/training.done
	@echo "📊 Evaluating MM-extra $* variant"
	@python scripts/mm_eval.py $* > $@
	@echo "✅ Evaluation completed for MM-extra $*"

# MM-extra Metrics 4 & 5 evaluation targets
trainings/MM/MM-extra/%-s42/mm_metrics_4_5.json: trainings/MM/MM-extra/%-s42/training.done
	@echo "🔬 Running Metrics 4 & 5 for MM-extra $* variant"
	@python scripts/mm_eval_transition_matrix.py \
		--transformer_path trainings/MM/MM-extra/$*-s42/ckpt.pt \
		--formal_model_path data/mm100/mm100-model.pkl \
		--output $@
	@echo "✅ Metrics 4 & 5 completed for MM-extra $*"

# MM-extra convenience targets (regular)
MM-extra-%: trainings/MM/MM-100-extra/MM-100-extra-%-s42/training.done
	@echo "✅ MM-extra $* completed via pattern rule"

MM-extra-all: $(addprefix trainings/MM/MM-100-extra/MM-100-extra-,$(addsuffix -s42/training.done,$(EXTRA_VARIANTS)))

# MM-extra convenience targets (sparse)
MM-100-sparse-75-extra-%: trainings/MM/MM-100-sparse-75-extra/MM-100-sparse-75-extra-%-s42/training.done
	@echo "✅ MM-100-sparse-75-extra $* completed via pattern rule"

MM-100-sparse-75-extra-all: $(addprefix trainings/MM/MM-100-sparse-75-extra/MM-100-sparse-75-extra-,$(addsuffix -s42/training.done,$(EXTRA_VARIANTS)))

# Sparse MM-100 evaluation convenience targets
eval-MM-100-sparse-75: trainings/MM/MM-100-sparse-75/eval_summary.txt
	@echo "✅ MM-100-sparse-75 evaluation completed"

metrics45-MM-100-sparse-75: $(addprefix trainings/MM/MM-100-sparse-75/MM-100-sparse-75-,$(addsuffix /mm_metrics_4_5.json,$(SEEDS)))
	@echo "✅ MM-100-sparse-75 Metrics 4 & 5 completed for all seeds"

# Sparse MM-extra evaluation convenience targets
eval-MM-100-sparse-75-extra-%: trainings/MM/MM-100-sparse-75-extra/MM-100-sparse-75-extra-%-s42/model_eval.txt
	@echo "✅ MM-100-sparse-75-extra $* evaluation completed"

eval-MM-100-sparse-75-extra-all: $(addprefix trainings/MM/MM-100-sparse-75-extra/MM-100-sparse-75-extra-,$(addsuffix -s42/model_eval.txt,$(EXTRA_VARIANTS)))
	@echo "✅ All MM-100-sparse-75-extra evaluations completed"

metrics45-MM-100-sparse-75-extra-%: trainings/MM/MM-100-sparse-75-extra/MM-100-sparse-75-extra-%-s42/mm_metrics_4_5.json
	@echo "✅ MM-100-sparse-75-extra $* Metrics 4 & 5 completed"

metrics45-MM-100-sparse-75-extra-all: $(addprefix trainings/MM/MM-100-sparse-75-extra/MM-100-sparse-75-extra-,$(addsuffix -s42/mm_metrics_4_5.json,$(EXTRA_VARIANTS)))
	@echo "✅ All MM-100-sparse-75-extra Metrics 4 & 5 completed"

# MM-extra evaluation targets
eval-MM-extra-%: trainings/MM/MM-extra/%-s42/model_eval.txt
	@echo "✅ MM-extra $* evaluation completed"

eval-MM-extra-all: $(addprefix trainings/MM/MM-extra/,$(addsuffix -s42/model_eval.txt,$(EXTRA_VARIANTS)))

# MM-extra Metrics 4 & 5 targets
metrics45-MM-extra-%: trainings/MM/MM-extra/%-s42/mm_metrics_4_5.json
	@echo "✅ MM-extra $* Metrics 4 & 5 completed"

metrics45-MM-extra-all: $(addprefix trainings/MM/MM-extra/,$(addsuffix -s42/mm_metrics_4_5.json,$(EXTRA_VARIANTS)))

# MM scale Metrics 4 & 5 targets  
metrics45-MM-100: trainings/MM/MM-100/MM-100-42/mm_metrics_4_5.json trainings/MM/MM-100/MM-100-123/mm_metrics_4_5.json trainings/MM/MM-100/MM-100-999/mm_metrics_4_5.json
	@echo "✅ MM-100 Metrics 4 & 5 completed for all seeds"

metrics45-MM-1000: trainings/MM/MM-1000/MM-1000-42/mm_metrics_4_5.json trainings/MM/MM-1000/MM-1000-123/mm_metrics_4_5.json trainings/MM/MM-1000/MM-1000-999/mm_metrics_4_5.json
	@echo "✅ MM-1000 Metrics 4 & 5 completed for all seeds"

metrics45-MM-%: $(addprefix trainings/MM/MM-$*/MM-$*-,$(addsuffix /mm_metrics_4_5.json,$(SEEDS)))
	@echo "✅ MM-$* Metrics 4 & 5 completed for all seeds"

# MM-extra summary evaluation
trainings/MM/MM-extra/eval_summary.txt: $(addprefix trainings/MM/MM-extra/,$(addsuffix -s42/model_eval.txt,$(EXTRA_VARIANTS)))
	@echo "📈 Generating MM-extra evaluation summary"
	@echo "# MM-extra Evaluation Summary" > $@
	@echo "Generated: $$(date)" >> $@
	@echo "" >> $@
	@for variant in $(EXTRA_VARIANTS); do \
		echo "## Variant $$variant" >> $@; \
		cat trainings/MM/MM-extra/$$variant-s42/model_eval.txt >> $@; \
		echo "" >> $@; \
	done
	@echo "✅ MM-extra evaluation summary completed"

# Comprehensive metrics aggregation targets
trainings/MM/MM-100-sparse-75/comprehensive_metrics.csv: trainings/MM/MM-100-sparse-75/eval_summary.txt $(addprefix trainings/MM/MM-100-sparse-75/MM-100-sparse-75-,$(addsuffix /mm_metrics_4_5.json,$(SEEDS)))
	@echo "📊 Generating comprehensive metrics for MM-100-sparse-75"
	@python scripts/aggregate_metrics.py \
		--base_dir trainings/MM/MM-100-sparse-75 \
		--output $@
	@echo "✅ Comprehensive metrics completed for MM-100-sparse-75"

trainings/MM/MM-100-sparse-75-extra/comprehensive_metrics.csv: $(addprefix trainings/MM/MM-100-sparse-75-extra/MM-100-sparse-75-extra-,$(addsuffix -s42/mm_metrics_4_5.json,$(EXTRA_VARIANTS)))
	@echo "📊 Generating comprehensive metrics for MM-100-sparse-75-extra"
	@python scripts/aggregate_metrics.py \
		--base_dir trainings/MM/MM-100-sparse-75-extra \
		--output $@
	@echo "✅ Comprehensive metrics completed for MM-100-sparse-75-extra"

trainings/MM/sparse_vs_dense_comparison.csv: trainings/MM/MM-100-sparse-75/comprehensive_metrics.csv trainings/MM/MM-100-sparse-75-extra/comprehensive_metrics.csv
	@echo "📊 Generating sparse vs dense comparison report"
	@echo "Dense MM-100 Results,Sparse MM-100 Results,Dense MM-extra Results,Sparse MM-extra Results" > $@
	@echo "✅ Sparse vs dense comparison placeholder created (manual analysis required)"

# Sparse MM report generation
trainings/MM/sparse_mm100_evaluation_report.md: trainings/MM/MM-100-sparse-75/comprehensive_metrics.csv trainings/MM/MM-100-sparse-75-extra/comprehensive_metrics.csv
	@echo "📊 Generating comprehensive sparse MM-100 evaluation report"
	@python scripts/generate_sparse_report.py --scale 100
	@echo "✅ Sparse MM-100 report generated: trainings/MM/sparse_mm100_evaluation_report.md"

# Complete pipeline targets
MM-100-sparse-75-FULL: trainings/MM/sparse_vs_dense_comparison.csv trainings/MM/sparse_mm100_evaluation_report.md
	@echo "🎉 Complete MM-100-sparse-75 pipeline finished!"
	@echo "📊 Results available in:"
	@echo "  - trainings/MM/MM-100-sparse-75/comprehensive_metrics.csv"
	@echo "  - trainings/MM/MM-100-sparse-75-extra/comprehensive_metrics.csv"
	@echo "  - trainings/MM/sparse_vs_dense_comparison.csv"
	@echo "  - trainings/MM/sparse_mm100_evaluation_report.md"

MM-100-FULL: trainings/MM/MM-100/eval_summary.txt trainings/MM/MM-extra/eval_summary.txt
	@echo "🎉 Complete MM-100 pipeline finished!"

# Convenience targets
sparse-comprehensive: trainings/MM/MM-100-sparse-75/comprehensive_metrics.csv
sparse-extra-comprehensive: trainings/MM/MM-100-sparse-75-extra/comprehensive_metrics.csv
sparse-comparison: trainings/MM/sparse_vs_dense_comparison.csv

# Help target
help:
	@echo "MM Framework - Available Targets:"
	@echo "================================"
	@echo "📖 Full documentation: MM/README.md"
	@echo ""
	@echo "  MM-10              Train MM-10 (sequential)"
	@echo "  MM-100             Train MM-100"  
	@echo "  MM-1000            Train MM-1000"
	@echo "  MM-100-sparse-75   Train MM-100 with 75% sparsity"
	@echo "  MM-extra-c1        Train MM-extra c1 (1L2H256D)"
	@echo "  MM-extra-c2        Train MM-extra c2 (1L1H128D)"
	@echo "  MM-extra-c3        Train MM-extra c3 (1L1H101D)"
	@echo "  MM-extra-c4        Train MM-extra c4 (1L1H101D+one-hot)"
	@echo "  MM-extra-c5        Train MM-extra c5 (1L MLP-only+one-hot)"
	@echo "  MM-extra-c6        Train MM-extra c6 (1L MLP-only+one-hot+1x-ratio)"
	@echo "  MM-extra-c7        Train MM-extra c7 (c4+pos_ape=False)"
	@echo "  MM-extra-c8        Train MM-extra c8 (c4+no LayerNorm)"
	@echo "  MM-extra-c9        Train MM-extra c9 (c4+no LayerNorm+no pos embeddings)"
	@echo "  MM-extra-c10       Train MM-extra c10 (c9+identity_first)"
	@echo "  MM-extra-c11       Train MM-extra c11 (c9+mlp_only)"
	@echo "  MM-extra-c12       Train MM-extra c12 (c10+mlp_only)"
	@echo "  MM-extra-all       Train all MM-extra variants"
	@echo "  MM-100-sparse-75-extra-c12  Train sparse c12 variant"
	@echo "  MM-100-sparse-75-extra-all  Train all sparse MM-extra variants"
	@echo "  eval-MM-extra-c1   Evaluate MM-extra c1"
	@echo "  eval-MM-extra-c2   Evaluate MM-extra c2"
	@echo "  eval-MM-extra-c3   Evaluate MM-extra c3"
	@echo "  eval-MM-extra-c4   Evaluate MM-extra c4"
	@echo "  eval-MM-extra-c5   Evaluate MM-extra c5"
	@echo "  eval-MM-extra-c6   Evaluate MM-extra c6"
	@echo "  eval-MM-extra-c7   Evaluate MM-extra c7"
	@echo "  eval-MM-extra-c8   Evaluate MM-extra c8"
	@echo "  eval-MM-extra-c9   Evaluate MM-extra c9"
	@echo "  eval-MM-extra-c10  Evaluate MM-extra c10"
	@echo "  eval-MM-extra-c11  Evaluate MM-extra c11"
	@echo "  eval-MM-extra-c12  Evaluate MM-extra c12"
	@echo "  eval-MM-extra-all  Evaluate all MM-extra variants"
	@echo "  eval-MM-100-sparse-75     Evaluate sparse MM-100 (all seeds)"
	@echo "  eval-MM-100-sparse-75-extra-c12  Evaluate sparse c12 variant"
	@echo "  eval-MM-100-sparse-75-extra-all  Evaluate all sparse MM-extra variants"
	@echo "  metrics45-MM-extra-c1   Run Metrics 4 & 5 for MM-extra c1"
	@echo "  metrics45-MM-extra-all  Run Metrics 4 & 5 for all MM-extra variants"
	@echo "  metrics45-MM-100        Run Metrics 4 & 5 for MM-100 (all seeds)"
	@echo "  metrics45-MM-1000       Run Metrics 4 & 5 for MM-1000 (all seeds)"
	@echo "  metrics45-MM-100-sparse-75  Run Metrics 4 & 5 for sparse MM-100"
	@echo "  metrics45-MM-100-sparse-75-extra-all  Run Metrics 4 & 5 for all sparse MM-extra"
	@echo "  MM-100-sparse-75-FULL       Complete sparse pipeline: train → eval → reports"
	@echo "  MM-100-FULL                 Complete dense pipeline: train → eval → reports"
	@echo "  sparse-comprehensive        Generate comprehensive sparse MM metrics"
	@echo "  sparse-extra-comprehensive  Generate comprehensive sparse MM-extra metrics"
	@echo "  sparse-comparison           Generate sparse vs dense comparison report"
	@echo ""
	@echo "Report Generation:"
	@echo "  trainings/MM/sparse_mm100_evaluation_report.md    Generate publication-ready sparse MM-100 report"
	@echo "  status             Show current progress"
	@echo "  clean-MM-10        Clean MM-10 results"
	@echo "  clean-MM-100       Clean MM-100 results" 
	@echo "  clean-MM-1000      Clean MM-1000 results"
	@echo "  clean-all          Clean all results"
	@echo ""
	@echo "Complete Pipelines:"
	@echo "  make MM-100-sparse-75-FULL    # Full sparse pipeline: data → train → eval → compare"
	@echo "  make MM-100-FULL              # Full dense pipeline: train → eval → summarize"
	@echo ""
	@echo "Parallel Execution (use -j flag):"
	@echo "  make -j4 MM-100-sparse-75-FULL   # Full pipeline with 4 parallel jobs"
	@echo "  make -j4 MM-100-sparse-75         # Train sparse MM with parallelism"
	@echo "  make -j4 MM-extra-all             # Train all MM-extra variants in parallel"
	@echo ""
	@echo "Examples:"
	@echo "  make MM-10                        # Train MM-10"
	@echo "  make MM-extra-c1                  # Train single MM-extra variant"
	@echo "  make status                   # Check progress"

.PHONY: all status clean-MM-% clean-all help MM-extra-% eval-MM-extra-% MM-extra-all eval-MM-extra-all metrics45-MM-% metrics45-MM-extra-% metrics45-MM-extra-all MM-100-sparse-75-FULL MM-100-FULL sparse-comprehensive sparse-extra-comprehensive sparse-comparison

# Default target when no arguments provided
.DEFAULT_GOAL := help
