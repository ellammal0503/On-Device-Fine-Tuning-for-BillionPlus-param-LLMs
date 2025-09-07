package com.pocketlora.scheduler

class TrainingScheduler {
    // TODO: Check battery, charging, and thermal signals
    fun canTrain(): Boolean {
        // Only train if charging, Wi-Fi on, device cool
        return true
    }
}
