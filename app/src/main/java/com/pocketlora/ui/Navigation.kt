package com.pocketlora.ui

import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.List
import androidx.compose.material.icons.filled.PlayArrow
import androidx.compose.material.icons.filled.Train
import androidx.compose.material3.Icon
import androidx.compose.material3.NavigationBar
import androidx.compose.material3.NavigationBarItem
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.navigation.NavDestination
import androidx.navigation.NavDestination.Companion.hierarchy
import androidx.navigation.NavHostController
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.currentBackStackEntryAsState
import androidx.navigation.compose.rememberNavController

sealed class Screen(val route: String, val label: String, val icon: @Composable () -> Unit) {
    object Training : Screen("training", "Training", { Icon(Icons.Filled.Train, "Training") })
    object Adapters : Screen("adapters", "Adapters", { Icon(Icons.Filled.List, "Adapters") })
    object Inference : Screen("inference", "Inference", { Icon(Icons.Filled.PlayArrow, "Inference") })
}

val screens = listOf(Screen.Training, Screen.Adapters, Screen.Inference)

@Composable
fun PocketLoRANav() {
    val navController = rememberNavController()
    androidx.compose.material3.Scaffold(
        bottomBar = { BottomBar(navController) }
    ) { innerPadding ->
        NavHost(
            navController,
            startDestination = Screen.Training.route,
            modifier = androidx.compose.ui.Modifier.padding(innerPadding)
        ) {
            composable(Screen.Training.route) { TrainingScreen() }
            composable(Screen.Adapters.route) { AdapterScreen() }
            composable(Screen.Inference.route) { InferenceScreen() }
        }
    }
}

@Composable
fun BottomBar(navController: NavHostController) {
    val navBackStackEntry by navController.currentBackStackEntryAsState()
    val currentDestination = navBackStackEntry?.destination
    NavigationBar {
        screens.forEach { screen ->
            NavigationBarItem(
                icon = { screen.icon() },
                label = { Text(screen.label) },
                selected = currentDestination.isTopLevelDestinationInHierarchy(screen),
                onClick = {
                    navController.navigate(screen.route) {
                        popUpTo(navController.graph.startDestinationId) { saveState = true }
                        launchSingleTop = true
                        restoreState = true
                    }
                }
            )
        }
    }
}

private fun NavDestination?.isTopLevelDestinationInHierarchy(screen: Screen) =
    this?.hierarchy?.any { it.route == screen.route } == true
