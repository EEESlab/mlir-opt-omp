# PULP-SDK app that links the test.o produced by run.sh into a runnable
# executable for gvsoc. Requires the GAP SDK environment to be sourced
# (RULES_DIR, toolchain, gvsoc). Driven by run.sh.
APP = test
APP_SRCS += pulp_main.c cluster_main.c interface-adapter.c
APP_CFLAGS += -O3
APP_LDFLAGS += -O3 test.o -lm

include $(RULES_DIR)/pmsis_rules.mk
