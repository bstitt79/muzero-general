import numpy
import backtrader as bt
from IPython.display import display, Image

from datetime import datetime

import btdata
import btstrategy

class PriceSizer(bt.Sizer):
    params = (('order_price', 20),)

    # this needs to return stake
    def _getsizing(self, comminfo, cash, data, isbuy):
        if data.close[0] == 0.0:
            return 0
        else:
            return max(int(self.p.order_price / data.close[0]), 1)

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
    def __init__(self, select_action_fn, stop_fn):
        self.select_action_fn = select_action_fn
        self.stop_fn = stop_fn

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
        cerebro.addstrategy(btstrategy.BTStrategy, self.select_action_fn, self.stop_fn, drawdown_call, order_price)

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
        ticker = numpy.random.choice(btdata.companies)
        
        while failing:

            print("trying ", ticker)
            
            data = bt.feeds.YahooFinanceData(dataname=ticker,
                                             fromdate=datetime(2017, 1, 1),
                                             todate=datetime(2020, 10, 1))
            
            cerebro.adddata(data) # modify this to support appropriate episode length

            try:
                failing = False
                result = cerebro.run()
            except FileNotFoundError as e:
                print(e)
                failing = True
                cerebro = self._init_cerebro()
                ticker = numpy.random.choice(btdata.companies)

        start_cash = cerebro.broker.startingcash
        final_cash = cerebro.broker.getvalue()
        unrealized_pnl = result[0].broker_stat['unrealized_pnl'][-1]

        reward = result[0].reward
        print("finished", ticker, ", final reward: ", reward)
        # this doesn't get used
        reward = (final_cash - start_cash - 1.1 * numpy.abs(unrealized_pnl)) / start_cash

        fig = cerebro.plot(style='line')[0][0]
        fig.set_size_inches(18,18)
        fig.savefig('plot.png', bbox_inches='tight', dpi=fig.dpi)
        display(Image(filename='plot.png'))

        return reward