TRAIN_LEVELS = [ "Sandstorm", "Dividers", "AllLockedUp", "ClosetSpace",
                "LeakInTheRoof", "2112GrandFinale", "WetSalmon", "RushingRiver",
                "CupOfWaffles", "Abandoned", "RunningMan", "Thirteen",
                "ReturnPostage", "Lines", "LiquidSuspension", "PizzaCat",
                "Foundation", "SimpleIsGood", "Stripes", "SectorThree",
                "SilentButDeadly2", "BlocksOfCheese", "CampbellsChickenNoodleSoup",
                "TwoFaced", "Temptation", "OverMyHead", "MyLife", "Thanksgiving",
                "Breif", "Superman", "Landskrona", "CambodianHoliday",
                "YumayaYOchun", "Convalescence", "Moose", "Mosquera", "Qualot",
                "PimP", "Equinox", "Jussipekka", "Prodigal", "Persim",
                "DoubleConfusion", "Pomeg", "Watmel", "Aidomok", "Cuchulainn",
                "MaputoExpress", "Hondew", "GossipCache", "RottenCore",
                "IMadeThisLevelDuringSchool", "MalteseFalcon", "Serenity",
                "Contentment", "TinMan", "Forever", "Superlock", "LOLevel",
                "BurnDownTheMission", "10by10", "Quickie", "YouMayBeRight", "Boo",
                "Misdirection", "NigerianWeatherForecast"]

TEST_LEVELS = ["K2xlgames", "Crazystylie", "LongWayHome", "Tetris",
               "AllBoxedUp", "TrickQuestion", "Corner2Corner", "HouseofGod",
               "RussianDoll", "Stars", "Mysterioso", "Brain", "Equality",
               "IndustrialBell", "OriginofSymmetry", "Puncture", "RabbitHole",
               "RoundtheBend", "StuckZipper", "Truancy", "TidalWave",
               "Enclosure", "Octopussy", "Checkmate", "Vortex", "TTotal",
               "SmallIntestines", "TrapDoor", "LinktoPast", "CrossEyed",
               "9by8"]

#Constant to determine how many shifts to keep when constructing
#data per position, in #expectation (assuming that at least this
#many shifts exist)
#Decrease this if memory consumed is too much. Increase it if data
#are insufficient
EXPECTED_SHIFTS_CHOSEN = .5

MAX_STEPS = 50

CPUT = 1.4142135

ILLEGAL = -1
NORMAL = 1
PUSH = 2
WIN = 10000

RAWPATH = "RawLevels/"
SOLVEDPATH = "SolvedLevels/"
SOLVABILITYPATH = "SolvabilityData/"

SIZE = 20