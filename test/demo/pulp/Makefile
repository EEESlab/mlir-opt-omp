# Self-contained PULP-SDK app that links the pmsis-lowered kernel.o (produced by
# run-pulp.sh) into a gvsoc-runnable GAP8 ELF and runs it.
#
# Needs the GAP SDK environment sourced so $(RULES_DIR)/pmsis_rules.mk resolves
# (run-pulp.sh sources PULP_SDK_ENV for you). Driven by run-pulp.sh:
#     make clean all run platform=gvsoc
APP = vecadd_pulp
APP_SRCS += pulp_main.c cluster_main.c interface-adapter.c
APP_CFLAGS += -O3
APP_LDFLAGS += -O3 kernel.o -lm

include $(RULES_DIR)/pmsis_rules.mk
