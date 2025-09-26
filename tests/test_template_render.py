import asyncio
from config import load_config
from core_app import CyclingAnalyzerApp

async def main():
    config = load_config()
    app = CyclingAnalyzerApp(config, test_mode=True)
    await app.initialize()
    activity_data = app.cache_manager.get("last_cycling_details")
    print("=== ACTIVITY DATA KEYS ===")
    print(list(activity_data.keys()) if activity_data else "No activity data")
    print("\nSample data:", dict(list(activity_data.items())[:10]) if activity_data else "No data")
    
    result = await app.analyze_workout("analyze_last_workout")
    print("=== TEST RESULT ===")
    print(result)
    await app.cleanup()

if __name__ == "__main__":
    asyncio.run(main())