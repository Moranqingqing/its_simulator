class Log:

    def __init__(self, logger):
        self.logger = logger

    def info(self, identifier, string):
        self.logger.info("{} {}".format(identifier, string))

    def warn(self, identifier, string):
        self.logger.warning("{} {}".format(identifier, string))
