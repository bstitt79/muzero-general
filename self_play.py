import math
import time

import numpy
import ray
import torch

import models

import backtrader as bt
from IPython.display import display, Image
from collections import deque
from datetime import datetime

class ReinformentLearningStrategy2(bt.Strategy):
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

class PriceSizer(bt.Sizer):
    params = (('order_price', 20),)

    # this needs to return stake
    def _getsizing(self, comminfo, cash, data, isbuy):
        if data.close[0] == 0.0:
            return 0
        else:
            return max(int(self.p.order_price / data.close[0]), 1)

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

class Reward(bt.observer.Observer):
    lines = ('reward',)
    plotinfo = dict(plot=True, subplot=True, plotname='Reward')
    plotlines = dict(reward=dict(markersize=4.0, color='darkviolet', fillstyle='full'))

    def next(self):
        self.lines.reward[0] = self._owner.reward

class BrokerStat(bt.observer.Observer):
    lines = ('cash', 'value',)
    plotinfo = dict(plot=True, subplot=True, plotname='$')

    def next(self):
        self.lines.cash[0] = self._owner.broker_stat['cash'][-1]
        self.lines.value[0] = self._owner.broker_stat['value'][-1]

class BrokerExposure(bt.observer.Observer):
    lines = ('position',)
    plotinfo = dict(plot=True, subplot=True, plotname='Position')
    plotlines = dict(position=dict(marker='.', markersize=1.0, color='blue', fillstyle='full'))

    def next(self):
        # assert self._owner.position.size == self._owner.broker_stat['exposure'][-1]
        # assert self._owner.position.size == sum([pos.size for pos in self._owner.positions.values()])
        self.lines.position[0] = self._owner.broker_stat['exposure'][-1]

class NormPnL(bt.observer.Observer):
    lines = ('realized_pnl', 'unrealized_pnl', 'max_unrealized_pnl', 'min_unrealized_pnl')
    plotinfo = dict(plot=True, subplot=True, plotname='PnL', plotymargin=.05)
    plotlines = dict(
        realized_pnl=dict(marker='o', markersize=4.0, color='blue', fillstyle='full'),
        unrealized_pnl=dict(marker='.', markersize=1.0, color='grey', fillstyle='full'),
        max_unrealized_pnl=dict(marker='.', markersize=1.0, color='c', fillstyle='full'),
        min_unrealized_pnl=dict(marker='.', markersize=1.0, color='m', fillstyle='full'),
    )

    def next(self):
        try:
            if self._owner.broker_stat['realized_pnl'][-1] != 0:
                self.lines.realized_pnl[0] = self._owner.broker_stat['realized_pnl'][-1]
        except IndexError:
            self.lines.realized_pnl[0] = 0.0

        try:
            self.lines.unrealized_pnl[0] = self._owner.broker_stat['unrealized_pnl'][-1]
        except IndexError:
            self.lines.unrealized_pnl[0] = 0.0

        try:
            self.lines.max_unrealized_pnl[0] = self._owner.broker_stat['max_unrealized_pnl'][-1]
            self.lines.min_unrealized_pnl[0] = self._owner.broker_stat['min_unrealized_pnl'][-1]

        except (IndexError, KeyError):
            self.lines.max_unrealized_pnl[0] = 0.0
            self.lines.min_unrealized_pnl[0] = 0.0

class BTEpisode():
    def __init__(self, select_action_fn, stop_fn, companies=['036570.KS']):
        self.select_action_fn = select_action_fn
        self.stop_fn = stop_fn
        self.companies = companies
        self._fig = None

    def _init_cerebro(self):
        start_cash = 1000000
        drawdown_call = 90
        commission = .0001 # = .01%
        slip_perc = 0.0 # 0.001 = .1%
        drawdown_amount = drawdown_call * start_cash / 100
        max_orders = 5
        order_price = drawdown_amount / max_orders

        cerebro = bt.Cerebro(
            stdstats=False
        )
        
        broker = bt.brokers.BackBroker(
            shortcash=False
        )
        broker.set_slippage_perc(slip_perc)
        broker.setcash(start_cash)
        broker.setcommission(commission=commission, leverage=1)
        
        cerebro.setbroker(broker)
        cerebro.addsizer(PriceSizer, order_price=order_price)

        # need model to persist across episodes and cerebros
        cerebro.addstrategy(ReinformentLearningStrategy2, self.select_action_fn, self.stop_fn, drawdown_call, order_price)

        cerebro.addobserver(bt.observers.BuySell)
        cerebro.addobserver(BrokerStat)
        cerebro.addobserver(BrokerExposure)
        cerebro.addobserver(bt.observers.DrawDown)
        cerebro.addobserver(bt.observers.Trades)
        cerebro.addobserver(NormPnL)
        cerebro.addobserver(Reward)

        return cerebro

    def render(self):
        if self._fig == None:
            print("No fig to render")
        else:
            self._fig.set_size_inches(18,18)
            self._fig.savefig('plot.png', bbox_inches='tight', dpi=fig.dpi)
            display(Image(filename='plot.png'))

    def run(self):
        failing = True
        cerebro = self._init_cerebro()
        
        while failing:
            ticker = numpy.random.choice(self.companies)

            print("trying ", ticker)
            
            data = bt.feeds.YahooFinanceData(dataname=ticker,
                                             fromdate=datetime(2017, 1, 1),
                                             todate=datetime(2020, 10, 1))
            
            cerebro.adddata(data) # modify this to support appropriate episode length

            try:
                failing = False
                result = cerebro.run()
            except FileNotFoundError as e:
                if companies == None:
                    raise
                print(e)
                failing = True
                cerebro = self._init_cerebro()

        start_cash = cerebro.broker.startingcash
        final_cash = cerebro.broker.getvalue()
        unrealized_pnl = result[0].broker_stat['unrealized_pnl'][-1]

        reward = (final_cash - start_cash - 1.1 * numpy.abs(unrealized_pnl)) / start_cash

        self._fig = cerebro.plot(style='line')[0][0]

        return reward

@ray.remote
class SelfPlay:
    """
    Class which run in a dedicated thread to play games and save them to the replay-buffer.
    """

    def __init__(self, initial_checkpoint, Game, config, seed):
        self.config = config
        self.game = Game(seed)

        # Fix random generator seed
        numpy.random.seed(seed)
        torch.manual_seed(seed)

        # Initialize the network
        self.model = models.MuZeroNetwork(self.config)
        self.model.set_weights(initial_checkpoint["weights"])
        self.model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.eval()

    def continuous_self_play(self, shared_storage, replay_buffer, test_mode=False):
        while ray.get(
            shared_storage.get_info.remote("training_step")
        ) < self.config.training_steps and not ray.get(
            shared_storage.get_info.remote("terminate")
        ):
            self.model.set_weights(ray.get(shared_storage.get_info.remote("weights")))

            if not test_mode:
                game_history = self.play_game(
                    self.config.visit_softmax_temperature_fn(
                        trained_steps=ray.get(
                            shared_storage.get_info.remote("training_step")
                        )
                    ),
                    self.config.temperature_threshold,
                    False,
                    "self",
                    0,
                )

                replay_buffer.save_game.remote(game_history, shared_storage)

            else:
                # Take the best action (no exploration) in test mode
                game_history = self.play_game(
                    0,
                    self.config.temperature_threshold,
                    False,
                    "self" if len(self.config.players) == 1 else self.config.opponent,
                    self.config.muzero_player,
                )

                # Save to the shared storage
                shared_storage.set_info.remote(
                    {
                        "episode_length": len(game_history.action_history) - 1,
                        "total_reward": sum(game_history.reward_history),
                        "mean_value": numpy.mean(
                            [value for value in game_history.root_values if value]
                        ),
                    }
                )
                if 1 < len(self.config.players):
                    shared_storage.set_info.remote(
                        {
                            "muzero_reward": sum(
                                reward
                                for i, reward in enumerate(game_history.reward_history)
                                if game_history.to_play_history[i - 1]
                                == self.config.muzero_player
                            ),
                            "opponent_reward": sum(
                                reward
                                for i, reward in enumerate(game_history.reward_history)
                                if game_history.to_play_history[i - 1]
                                != self.config.muzero_player
                            ),
                        }
                    )

            # Managing the self-play / training ratio
            if not test_mode and self.config.self_play_delay:
                time.sleep(self.config.self_play_delay)
            if not test_mode and self.config.ratio:
                while (
                    ray.get(shared_storage.get_info.remote("training_step"))
                    / max(
                        1, ray.get(shared_storage.get_info.remote("num_played_steps"))
                    )
                    < self.config.ratio
                    and ray.get(shared_storage.get_info.remote("training_step"))
                    < self.config.training_steps
                    and not ray.get(shared_storage.get_info.remote("terminate"))
                ):
                    time.sleep(0.5)

        self.close_game()

    def play_game(
        self, temperature, temperature_threshold, render, opponent, muzero_player
    ):
        """
        Play one game with actions based on the Monte Carlo tree search at each moves.
        """
        game_history = GameHistory()
        
        # observation = self.game.reset()
        self.tmp_observation = self.game.reset()
        self.tmp_action = 0
        self.tmp_reward = 0
        self.tmp_to_play = self.game.to_play()
        self.tmp_search_stats = None
        # should be one less search stat than all rest

        # game_history.action_history.append(0) # this is a side effect of game reset
        # game_history.observation_history.append(observation)
        # game_history.reward_history.append(0) # this is a side effect of game reset
        # game_history.to_play_history.append(self.game.to_play())

        done = False

        def select_action_fn(observation, reward):
            if self.tmp_search_stats != None:
                tmp_root, tmp_action_space = self.tmp_search_stats
                game_history.store_search_statistics(tmp_root, tmp_action_space)

            game_history.action_history.append(self.tmp_action)
            game_history.observation_history.append(observation)
            game_history.reward_history.append(reward)
            game_history.to_play_history.append(self.tmp_to_play)

            assert (
                len(numpy.array(observation).shape) == 3
            ), f"Observation should be 3 dimensionnal instead of {len(numpy.array(observation).shape)} dimensionnal. Got observation of shape: {numpy.array(observation).shape}"
            assert (
                numpy.array(observation).shape == self.config.observation_shape
            ), f"Observation should match the observation_shape defined in MuZeroConfig. Expected {self.config.observation_shape} but got {numpy.array(observation).shape}."
            stacked_observations = game_history.get_stacked_observations(
                -1,
                self.config.stacked_observations,
            )

            # Choose the action
            if opponent == "self" or muzero_player == self.game.to_play():
                root, mcts_info = MCTS(self.config).run(
                    self.model,
                    stacked_observations,
                    self.game.legal_actions(),
                    self.game.to_play(),
                    True,
                )
                action = self.select_action(
                    root,
                    temperature
                    if not temperature_threshold
                    or len(game_history.action_history) < temperature_threshold
                    else 0,
                )

                if render:
                    print(f'Tree depth: {mcts_info["max_tree_depth"]}')
                    print(
                        f"Root value for player {self.game.to_play()}: {root.value():.2f}"
                    )
            else:
                action, root = self.select_opponent_action(
                    opponent, stacked_observations
                )

            # observation, reward, done = self.game.step(action)

            self.tmp_action = action
            # self.tmp_observation = observation
            # self.tmp_reward = reward
            self.tmp_to_play = self.game.to_play()
            self.tmp_search_stats = root, self.config.action_space

            return action

        def stop_fn(final_reward):
            game_history.reward_history[-1] = final_reward

        episode = BTEpisode(
            select_action_fn,
            stop_fn
        )

        with torch.no_grad():
            episode.run()

        if render:
            episode.render()

        return game_history

    def close_game(self):
        self.game.close()

    def select_opponent_action(self, opponent, stacked_observations):
        """
        Select opponent action for evaluating MuZero level.
        """
        if opponent == "human":
            root, mcts_info = MCTS(self.config).run(
                self.model,
                stacked_observations,
                self.game.legal_actions(),
                self.game.to_play(),
                True,
            )
            print(f'Tree depth: {mcts_info["max_tree_depth"]}')
            print(f"Root value for player {self.game.to_play()}: {root.value():.2f}")
            print(
                f"Player {self.game.to_play()} turn. MuZero suggests {self.game.action_to_string(self.select_action(root, 0))}"
            )
            return self.game.human_to_action(), root
        elif opponent == "expert":
            return self.game.expert_agent(), None
        elif opponent == "random":
            assert (
                self.game.legal_actions()
            ), f"Legal actions should not be an empty array. Got {self.game.legal_actions()}."
            assert set(self.game.legal_actions()).issubset(
                set(self.config.action_space)
            ), "Legal actions should be a subset of the action space."

            return numpy.random.choice(self.game.legal_actions()), None
        else:
            raise NotImplementedError(
                'Wrong argument: "opponent" argument should be "self", "human", "expert" or "random"'
            )

    @staticmethod
    def select_action(node, temperature):
        """
        Select action according to the visit count distribution and the temperature.
        The temperature is changed dynamically with the visit_softmax_temperature function
        in the config.
        """
        visit_counts = numpy.array(
            [child.visit_count for child in node.children.values()], dtype="int32"
        )
        actions = [action for action in node.children.keys()]
        if temperature == 0:
            action = actions[numpy.argmax(visit_counts)]
        elif temperature == float("inf"):
            action = numpy.random.choice(actions)
        else:
            # See paper appendix Data Generation
            visit_count_distribution = visit_counts ** (1 / temperature)
            visit_count_distribution = visit_count_distribution / sum(
                visit_count_distribution
            )
            action = numpy.random.choice(actions, p=visit_count_distribution)

        return action


# Game independent
class MCTS:
    """
    Core Monte Carlo Tree Search algorithm.
    To decide on an action, we run N simulations, always starting at the root of
    the search tree and traversing the tree according to the UCB formula until we
    reach a leaf node.
    """

    def __init__(self, config):
        self.config = config

    def run(
        self,
        model,
        observation,
        legal_actions,
        to_play,
        add_exploration_noise,
        override_root_with=None,
    ):
        """
        At the root of the search tree we use the representation function to obtain a
        hidden state given the current observation.
        We then run a Monte Carlo Tree Search using only action sequences and the model
        learned by the network.
        """
        if override_root_with:
            root = override_root_with
            root_predicted_value = None
        else:
            root = Node(0)
            observation = (
                torch.tensor(observation)
                .float()
                .unsqueeze(0)
                .to(next(model.parameters()).device)
            )
            (
                root_predicted_value,
                reward,
                policy_logits,
                hidden_state,
            ) = model.initial_inference(observation)
            root_predicted_value = models.support_to_scalar(
                root_predicted_value, self.config.support_size
            ).item()
            reward = models.support_to_scalar(reward, self.config.support_size).item()
            assert (
                legal_actions
            ), f"Legal actions should not be an empty array. Got {legal_actions}."
            assert set(legal_actions).issubset(
                set(self.config.action_space)
            ), "Legal actions should be a subset of the action space."
            root.expand(
                legal_actions,
                to_play,
                reward,
                policy_logits,
                hidden_state,
            )

        if add_exploration_noise:
            root.add_exploration_noise(
                dirichlet_alpha=self.config.root_dirichlet_alpha,
                exploration_fraction=self.config.root_exploration_fraction,
            )

        min_max_stats = MinMaxStats()

        max_tree_depth = 0
        for _ in range(self.config.num_simulations):
            virtual_to_play = to_play
            node = root
            search_path = [node]
            current_tree_depth = 0

            while node.expanded():
                current_tree_depth += 1
                action, node = self.select_child(node, min_max_stats)
                search_path.append(node)

                # Players play turn by turn
                if virtual_to_play + 1 < len(self.config.players):
                    virtual_to_play = self.config.players[virtual_to_play + 1]
                else:
                    virtual_to_play = self.config.players[0]

            # Inside the search tree we use the dynamics function to obtain the next hidden
            # state given an action and the previous hidden state
            parent = search_path[-2]
            value, reward, policy_logits, hidden_state = model.recurrent_inference(
                parent.hidden_state,
                torch.tensor([[action]]).to(parent.hidden_state.device),
            )
            value = models.support_to_scalar(value, self.config.support_size).item()
            reward = models.support_to_scalar(reward, self.config.support_size).item()
            node.expand(
                self.config.action_space,
                virtual_to_play,
                reward,
                policy_logits,
                hidden_state,
            )

            self.backpropagate(search_path, value, virtual_to_play, min_max_stats)

            max_tree_depth = max(max_tree_depth, current_tree_depth)

        extra_info = {
            "max_tree_depth": max_tree_depth,
            "root_predicted_value": root_predicted_value,
        }
        return root, extra_info

    def select_child(self, node, min_max_stats):
        """
        Select the child with the highest UCB score.
        """
        max_ucb = max(
            self.ucb_score(node, child, min_max_stats)
            for action, child in node.children.items()
        )
        action = numpy.random.choice(
            [
                action
                for action, child in node.children.items()
                if self.ucb_score(node, child, min_max_stats) == max_ucb
            ]
        )
        return action, node.children[action]

    def ucb_score(self, parent, child, min_max_stats):
        """
        The score for a node is based on its value, plus an exploration bonus based on the prior.
        """
        pb_c = (
            math.log(
                (parent.visit_count + self.config.pb_c_base + 1) / self.config.pb_c_base
            )
            + self.config.pb_c_init
        )
        pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)

        prior_score = pb_c * child.prior

        if child.visit_count > 0:
            # Mean value Q
            value_score = min_max_stats.normalize(
                child.reward
                + self.config.discount
                * (child.value() if len(self.config.players) == 1 else -child.value())
            )
        else:
            value_score = 0

        return prior_score + value_score

    def backpropagate(self, search_path, value, to_play, min_max_stats):
        """
        At the end of a simulation, we propagate the evaluation all the way up the tree
        to the root.
        """
        if len(self.config.players) == 1:
            for node in reversed(search_path):
                node.value_sum += value
                node.visit_count += 1
                min_max_stats.update(node.reward + self.config.discount * node.value())

                value = node.reward + self.config.discount * value

        elif len(self.config.players) == 2:
            for node in reversed(search_path):
                node.value_sum += value if node.to_play == to_play else -value
                node.visit_count += 1
                min_max_stats.update(node.reward + self.config.discount * -node.value())

                value = (
                    -node.reward if node.to_play == to_play else node.reward
                ) + self.config.discount * value

        else:
            raise NotImplementedError("More than two player mode not implemented.")


class Node:
    def __init__(self, prior):
        self.visit_count = 0
        self.to_play = -1
        self.prior = prior
        self.value_sum = 0
        self.children = {}
        self.hidden_state = None
        self.reward = 0

    def expanded(self):
        return len(self.children) > 0

    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

    def expand(self, actions, to_play, reward, policy_logits, hidden_state):
        """
        We expand a node using the value, reward and policy prediction obtained from the
        neural network.
        """
        self.to_play = to_play
        self.reward = reward
        self.hidden_state = hidden_state

        policy_values = torch.softmax(
            torch.tensor([policy_logits[0][a] for a in actions]), dim=0
        ).tolist()
        policy = {a: policy_values[i] for i, a in enumerate(actions)}
        for action, p in policy.items():
            self.children[action] = Node(p)

    def add_exploration_noise(self, dirichlet_alpha, exploration_fraction):
        """
        At the start of each search, we add dirichlet noise to the prior of the root to
        encourage the search to explore new actions.
        """
        actions = list(self.children.keys())
        noise = numpy.random.dirichlet([dirichlet_alpha] * len(actions))
        frac = exploration_fraction
        for a, n in zip(actions, noise):
            self.children[a].prior = self.children[a].prior * (1 - frac) + n * frac


class GameHistory:
    """
    Store only usefull information of a self-play game.
    """

    def __init__(self):
        self.observation_history = []
        self.action_history = []
        self.reward_history = []
        self.to_play_history = []
        self.child_visits = []
        self.root_values = []
        self.reanalysed_predicted_root_values = None
        # For PER
        self.priorities = None
        self.game_priority = None

    def store_search_statistics(self, root, action_space):
        # Turn visit count from root into a policy
        if root is not None:
            sum_visits = sum(child.visit_count for child in root.children.values())
            self.child_visits.append(
                [
                    root.children[a].visit_count / sum_visits
                    if a in root.children
                    else 0
                    for a in action_space
                ]
            )

            self.root_values.append(root.value())
        else:
            self.root_values.append(None)

    def get_stacked_observations(self, index, num_stacked_observations):
        """
        Generate a new observation with the observation at the index position
        and num_stacked_observations past observations and actions stacked.
        """
        # Convert to positive index
        index = index % len(self.observation_history)

        stacked_observations = self.observation_history[index].copy()
        for past_observation_index in reversed(
            range(index - num_stacked_observations, index)
        ):
            if 0 <= past_observation_index:
                previous_observation = numpy.concatenate(
                    (
                        self.observation_history[past_observation_index],
                        [
                            numpy.ones_like(stacked_observations[0])
                            * self.action_history[past_observation_index + 1]
                        ],
                    )
                )
            else:
                previous_observation = numpy.concatenate(
                    (
                        numpy.zeros_like(self.observation_history[index]),
                        [numpy.zeros_like(stacked_observations[0])],
                    )
                )

            stacked_observations = numpy.concatenate(
                (stacked_observations, previous_observation)
            )

        return stacked_observations


class MinMaxStats:
    """
    A class that holds the min-max values of the tree.
    """

    def __init__(self):
        self.maximum = -float("inf")
        self.minimum = float("inf")

    def update(self, value):
        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)

    def normalize(self, value):
        if self.maximum > self.minimum:
            # We normalize only when we have set the maximum and minimum values
            return (value - self.minimum) / (self.maximum - self.minimum)
        return value
