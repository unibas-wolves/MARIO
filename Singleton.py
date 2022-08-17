class SingletonMeta(type):
    """
        The Singleton class can be implemented in different ways in Python. Some
        possible methods include: base class, decorator, metaclass. We will use the
        metaclass because it is best suited for this purpose.
    """

    _instances = {}

    def __call__(cls, *args, **kwargs):
        """
            Possible changes to the value of the `__init__` argument do not affect
            the returned instance.
        """
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]
        self.path_gc_team = ""
        self.path_gc_game = ""
        self.homeTeam = ""
        self.awayTeam = ""
        self.videoPath =""
        self.calibPath = ""

class Singleton(metaclass=SingletonMeta):
    
    def set_gc_team_path(self,path):
        self.path_gc_team = path

    def get_gc_team_path(self):
        return self.path_gc_team

    def set_gc_game_path(self,path):
        self.path_gc_game = path

    def get_gc_game_path(self):
        return self.path_gc_game

    def setHomeTeam(self,team):
        self.homeTeam = team

    def getHomeTeam(self):
        return self.homeTeam

    def setAwayTeam(self,team):
        self.awayTeam = team

    def getAwayTeam(self):
        return self.awayTeam

    def setVideoPath(self,path):
        self.videoPath = path

    def getVideoPath(self):
        return self.videoPath

    def setCalibPath(self,path):
        self.calibPath = path

    def getCalibPath(self):
        return self.calibPath


