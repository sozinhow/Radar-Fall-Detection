################################################################################
# Automatically-generated file. Do not edit!
# Toolchain: GNU Tools for STM32 (13.3.rel1)
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../AI/App/network.c \
../AI/App/network_data.c 

OBJS += \
./AI/App/network.o \
./AI/App/network_data.o 

C_DEPS += \
./AI/App/network.d \
./AI/App/network_data.d 


# Each subdirectory must supply rules for building sources it contributes
AI/App/%.o AI/App/%.su AI/App/%.cyclo: ../AI/App/%.c AI/App/subdir.mk
	arm-none-eabi-gcc "$<" -mcpu=cortex-m7 -std=gnu11 -g3 -DDEBUG -DUSE_HAL_DRIVER -DSTM32F746xx -c -I../Core/Inc -I../Drivers/STM32F7xx_HAL_Driver/Inc -I../Drivers/STM32F7xx_HAL_Driver/Inc/Legacy -I../Drivers/CMSIS/Device/ST/STM32F7xx/Include -I../Drivers/CMSIS/Include -I../Middlewares/ST/AI/Inc -I../AI/App -O0 -ffunction-sections -fdata-sections -Wall -fstack-usage -fcyclomatic-complexity -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" --specs=nano.specs -mfpu=fpv5-sp-d16 -mfloat-abi=hard -mthumb -o "$@"

clean: clean-AI-2f-App

clean-AI-2f-App:
	-$(RM) ./AI/App/network.cyclo ./AI/App/network.d ./AI/App/network.o ./AI/App/network.su ./AI/App/network_data.cyclo ./AI/App/network_data.d ./AI/App/network_data.o ./AI/App/network_data.su

.PHONY: clean-AI-2f-App

