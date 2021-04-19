from collections import deque
import numpy
import backtrader as bt

class Differencing(bt.Indicator):
    alias = ('Differencing', 'Dif')
    lines = ('dif',)

    plotinfo = dict(
        plot=True,
        subplot=True,
        plotname='+/-')

    def __init__(self):
        self.addminperiod(1)

    def nextstart(self):
        self.l.dif[0] = 0  # seed value

    def next(self):
        self.l.dif[0] = self.data[0] - self.data[-1]

class KalmanMovingAverage(bt.Indicator):
    packages = ('pykalman',)
    frompackages = (('pykalman', [('KalmanFilter', 'KF')]),)
    plotinfo = dict(plot=True, subplot=True, plotname='kal')
    lines = ('kma',)
    alias = ('KMA',)
    params = (
        ('initial_state_covariance', 1.0),
        ('observation_covariance', 1.0),
        ('transition_covariance', 0.05),
    )

    def __init__(self, obs_cov=1.0, trans_cov=.05):
        self.p.observation_covariance = obs_cov
        self.p.transition_covariance = trans_cov
        self._dlast = self.data(-1)  # get previous day value, this creates a new line object delayed by 1

    def nextstart(self):
        self._k1 = self._dlast[0]
        self._c1 = self.p.initial_state_covariance

        self._kf = pykalman.KalmanFilter(
            transition_matrices=[1],
            observation_matrices=[1],
            observation_covariance=self.p.observation_covariance,
            transition_covariance=self.p.transition_covariance,
            initial_state_mean=self._k1,
            initial_state_covariance=self._c1,
        )

        self.next()

    def next(self):
        k1, self._c1 = self._kf.filter_update(self._k1, self._c1, self.data[0])
        self.lines.kma[0] = self._k1 = k1

class BTStrategy(bt.Strategy):
    # action_picker = (obs, reward) -> action
    def __init__(self, action_selector_fn, stop_fn, drawdown_call=10, order_price=20):
        # this is called before strat is added to cerebro
        # cerebro is built and run every episode, aka this only needs to work for one episode

        self.action_selector_fn = action_selector_fn
        self.stop_fn = stop_fn

        self.rewards = []

        self.time_dim = 2 # need two for delta calculations

        self.order_price = order_price
        self.drawdown_call = drawdown_call
        self.target_call = drawdown_call

        self.trade_just_closed = False
        self.trade_result = 0

        self.unrealized_pnl = None
        self.norm_broker_value = None
        self.realized_pnl = None

        self.current_pos_duration = 0
        self.current_pos_min_value = 0
        self.current_pos_max_value = 0

        # seperate option
        # add slow, medium, fast kalmans
        # add kalman f, m, s (these will end up being small because dif open is centered at 0, I think that is ok)
        # test these with hyper params

        self.data.open_norm = Differencing(self.data.open)
        self.kalof = KalmanMovingAverage(self.data.open_norm, trans_cov=.1)
        self.kalom = KalmanMovingAverage(self.data.open_norm, trans_cov=.001)
        self.kalos = KalmanMovingAverage(self.data.open_norm, trans_cov=.00001)
        
        self.data.high_norm = Differencing(self.data.high)
        self.kalhf = KalmanMovingAverage(self.data.high_norm, trans_cov=.1)
        self.kalhm = KalmanMovingAverage(self.data.high_norm, trans_cov=.001)
        self.kalhs = KalmanMovingAverage(self.data.high_norm, trans_cov=.00001)
        
        self.data.low_norm = Differencing(self.data.low)
        self.kallf = KalmanMovingAverage(self.data.low_norm, trans_cov=.1)
        self.kallm = KalmanMovingAverage(self.data.low_norm, trans_cov=.001)
        self.kalls = KalmanMovingAverage(self.data.low_norm, trans_cov=.00001)
        
        self.data.close_norm = Differencing(self.data.close)
        self.kalcf = KalmanMovingAverage(self.data.close_norm, trans_cov=.1)
        self.kalcm = KalmanMovingAverage(self.data.close_norm, trans_cov=.001)
        self.kalcs = KalmanMovingAverage(self.data.close_norm, trans_cov=.00001)
        
        # Service sma to get correct first features values:
        self.data.dim_sma = bt.indicators.SimpleMovingAverage( # does this cause the blocking for period?
            self.datas[0],
            period=self.time_dim
        )
        self.data.dim_sma.plotinfo.plot = False

        self.target_value = self.env.broker.startingcash * (1 + self.target_call / 100)
        self.drawdown_value = self.env.broker.startingcash - self.drawdown_call * self.env.broker.startingcash / 100
        self.reward = 0.

        self.realized_broker_value = self.env.broker.startingcash

        self.broker_datalines = [
            'cash',
            'value',
            'exposure',
            'drawdown',
            'realized_pnl',
            'unrealized_pnl',
            'min_unrealized_pnl',
            'max_unrealized_pnl',
        ]

        self.collection_get_broker_stat_methods = {}
        for line in self.broker_datalines:
            try:
                self.collection_get_broker_stat_methods[line] = getattr(self, 'get_broker_{}'.format(line))

            except AttributeError:
                raise NotImplementedError('Callable get_broker_{}.() not found'.format(line))

        self.broker_stat = {key: deque(maxlen=self.time_dim) for key in self.broker_datalines}

    def notify_trade(self, trade):
        if trade.isclosed:
            self.trade_just_closed = True
            self.trade_result = trade.pnlcomm

            # Store realized prtfolio value:
            self.realized_broker_value = self.env.broker.get_value()

            # self.log('OPERATION PROFIT, GROSS %.2f, NET %.2f' %
            #     (trade.pnl, trade.pnlcomm))

    def execute(self, action):
        if action == 0: # hold
            pass
        elif action == 1: # buy
            self.buy()
        elif action == 2: # sell
            self.sell()
        elif action == 3: # close
            self.close()

    # is this useful?
    def prenext(self):
        self.update_broker_stat()

    # does this get used?
    def nextstart(self):
        self.inner_embedding = self.data.close.buflen()
        # self.log.debug('Inner time embedding: {}'.format(self.inner_embedding))

    def next(self):
        self.update_broker_stat()

        reward = self.get_reward()

        observation = self.get_state()

        selected_action = self.action_selector_fn(observation, reward)

        self.execute(selected_action) # apply action, action must be last step in next, (obs, reward, done) is first step

        hit_drawdown = self.stats.drawdown.maxdrawdown[0] >= self.drawdown_call
        hit_target = self.env.broker.get_value() > self.target_value
        early_stop = hit_drawdown or hit_target

        if early_stop:
            self.env.runstop()

    def stop(self):
        final_reward = self.env.broker.get_value() - self.env.broker.startingcash - 1.1 * self.broker_stat['unrealized_pnl'][-1]
        self.stop_fn(final_reward)

    def get_state(self):
        # lines most recent = 0
        # deque most recent = -1
        observation = numpy.array([
                self.data.open_norm[0],
                self.kalof[0],
                self.kalom[0],
                self.kalos[0],
                self.data.high_norm[0],
                self.kalhf[0],
                self.kalhm[0],
                self.kalhs[0],
                self.data.low_norm[0],
                self.kallf[0],
                self.kallm[0],
                self.kalls[0],
                self.data.close_norm[0],
                self.kalcf[0],
                self.kalcm[0],
                self.kalcs[0],
                self.data.volume[0], # normalize volume, sometimes this returns 0
                self.broker_stat['value'][-1],
                self.broker_stat['cash'][-1],
                self.broker_stat['unrealized_pnl'][-1],
                self.broker_stat['realized_pnl'][-1],
                self.broker_stat['exposure'][-1],
                self.broker_stat['drawdown'][-1]
             ])

        return observation[numpy.newaxis, numpy.newaxis, :] # (1, 1, 23)
        
    def get_reward(self):
        current_pos_duration = self.get_broker_pos_duration()
        if current_pos_duration == 0:
            unrealized_pnl_delta = 0
        else:
            unrealized_pnl_delta = self.broker_stat['unrealized_pnl'][-1] - self.broker_stat['unrealized_pnl'][-2]
        
        realized_pnl = self.broker_stat['realized_pnl'][-1]

        self.reward = unrealized_pnl_delta + realized_pnl

        return self.reward

    def get_broker_pos_duration(self):
        if self.position.size == 0:
            self.current_pos_duration = 0
        else:
            self.current_pos_duration += 1
        return self.current_pos_duration

    def update_broker_stat(self):
        for key, method in self.collection_get_broker_stat_methods.items():
            self.broker_stat[key].append(method())

        # Reset one-time flags:
        self.trade_just_closed = False

    # value - [-1, 1] = 2 * (value - lower) / (upper - lower) - 1 where lower = drawdown, upper = target
    def get_broker_value(self):
        return 2 * (self.env.broker.getvalue() - self.drawdown_value) / (self.target_value - self.drawdown_value) - 1

    # cash - [-1, 1] = 2 * (value - lower) / (upper - lower) - 1 where lower = drawdown, upper = target    
    def get_broker_cash(self):
        return 2 * (self.env.broker.get_cash() - self.drawdown_value) / (self.target_value - self.drawdown_value) - 1

    def get_broker_exposure(self):
        return self.position.size        

    def get_broker_realized_pnl(self):
        if self.trade_just_closed:
            return self.trade_result
        else:
            return 0.0

    def get_broker_unrealized_pnl(self):
        return self.env.broker.get_value() - self.realized_broker_value

    def get_broker_drawdown(self):
        try:
            dd = numpy.nan_to_num(self.stats.drawdown.drawdown[-1]) / 100 # (this is then normed 0, 1)
        except IndexError:
            dd = 0.0

        return dd

    def get_broker_max_unrealized_pnl(self):
        current_value = self.env.broker.get_value()
        if self.position.size == 0:
            self.current_pos_max_value = current_value
        else:
            if self.current_pos_max_value < current_value:
                self.current_pos_max_value = current_value
        return (self.current_pos_max_value - self.realized_broker_value)

    def get_broker_min_unrealized_pnl(self):
        # reset this in notify trade
        current_value = self.env.broker.get_value()
        if self.position.size == 0:
            self.current_pos_min_value = current_value
        else:
            if self.current_pos_min_value > current_value:
                self.current_pos_min_value = current_value
        return (self.current_pos_min_value - self.realized_broker_value)