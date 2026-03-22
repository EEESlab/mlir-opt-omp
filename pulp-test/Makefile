APP = matmul
APP_SRCS += main.c interface-adapter.c

APP_CFLAGS += -DCLUSTER -DNUM_CLUSTER=1
APP_CFLAGS += -DNUM_CORES=8

APP_CFLAGS += -O3
APP_LDFLAGS += -O3 kernel.o

include $(RULES_DIR)/pmsis_rules.mk
