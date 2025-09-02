#!/usr/bin/env python3
"""
LSTM-based trading strategy for iron ore futures.

This module implements a momentum-based trading strategy using LSTM predictions:
- Entry: When predicted price difference exceeds $1.20 threshold
- Exit: When prediction direction reverses OR end-of-month (mandatory)
- Position: Multiple entries allowed, exit all positions simultaneously

Strategy Logic:
1. Calculate predicted price: current_price * (1 + predicted_ln_returns)
2. Price difference: predicted_price - current_price
3. Enter LONG if price_diff > $1.20, SHORT if price_diff < -$1.20
4. Exit all positions when direction reverses or month ends
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime
import logging
import json

logger = logging.getLogger(__name__)

# Trading constants
BUFFER_THRESHOLD = 0.30  #  absolute price difference threshold
POSITION_SIZE = 1.0  # Standard position size per signal


@dataclass
class Trade:
    """Simple trade tracking for position management."""

    entry_date: pd.Timestamp
    entry_price: float
    trade_type: str  # "LONG" or "SHORT"
    position_size: float
    exit_date: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    pnl: Optional[float] = None

    def is_open(self) -> bool:
        """Check if trade is still open."""
        return self.exit_date is None

    def close_trade(
        self, exit_date: pd.Timestamp, exit_price: float, exit_reason: str
    ) -> None:
        """Close the trade and calculate P&L."""
        self.exit_date = exit_date
        self.exit_price = exit_price
        self.exit_reason = exit_reason

        # Calculate P&L based on trade type
        if self.trade_type == "LONG":
            self.pnl = (exit_price - self.entry_price) * self.position_size
        else:  # SHORT
            self.pnl = (self.entry_price - exit_price) * self.position_size

    def to_dict(self) -> Dict[str, Any]:
        """Convert trade to dictionary for export."""
        return {
            "entry_date": self.entry_date,
            "entry_price": self.entry_price,
            "trade_type": self.trade_type,
            "position_size": self.position_size,
            "exit_date": self.exit_date,
            "exit_price": self.exit_price,
            "exit_reason": self.exit_reason,
            "pnl": self.pnl,
            "holding_days": (
                (self.exit_date - self.entry_date).days if self.exit_date else None
            ),
        }


class LSTMTradingStrategy:
    """LSTM-based momentum trading strategy for iron ore futures."""

    def __init__(self) -> None:
        """Initialize trading strategy."""
        self.trades: List[Trade] = []
        self.open_positions: List[Trade] = []
        self.daily_pnl: List[float] = []
        self.equity_curve: List[float] = []

        # Enhanced position tracking inspired by sample strategy
        self.daily_realized_pnl: List[float] = []  # P&L from closed trades only
        self.daily_unrealized_pnl: List[float] = []  # Mark-to-market on open positions
        self.daily_total_pnl: List[float] = []  # Realized + Unrealized
        self.daily_net_position: List[float] = []  # Net position size (LONG - SHORT)
        self.daily_gross_position: List[float] = []  # Total absolute position size

    def generate_signal(
        self, raw_price: float, predicted_ln_returns: float
    ) -> Optional[str]:
        """
        Generate trading signal based on LSTM prediction.

        Args:
            raw_price: Current raw 65% M+1 DSP price
            predicted_ln_returns: LSTM predicted log returns

        Returns:
            "LONG", "SHORT", or None
        """
        # Validate inputs to prevent spurious entries from NaN/Inf data
        if not np.isfinite(raw_price) or not np.isfinite(predicted_ln_returns):
            return None

        # Convert log returns to simple returns first (best practice)
        # Note: predicted_ln_returns are in percentage form (×100), so divide by 100 first
        simple_return = np.exp(predicted_ln_returns / 100) - 1

        # Calculate predicted price difference using simple returns
        price_difference = raw_price * simple_return

        # Apply buffer threshold
        if price_difference > BUFFER_THRESHOLD:
            return "LONG"
        elif price_difference < -BUFFER_THRESHOLD:
            return "SHORT"
        else:
            return None

    def should_exit_on_reversal(
        self, trade: Trade, predicted_ln_returns: float
    ) -> bool:
        """
        Check if position should exit due to direction reversal.

        Args:
            trade: Open trade position
            predicted_ln_returns: Current LSTM prediction

        Returns:
            True if should exit, False otherwise
        """
        if trade.trade_type == "LONG" and predicted_ln_returns <= 0:
            return True
        elif trade.trade_type == "SHORT" and predicted_ln_returns >= 0:
            return True
        return False

    def is_month_end(
        self, current_date: pd.Timestamp, next_date: Optional[pd.Timestamp]
    ) -> bool:
        """
        Check if current date is actual end of month (last trading day of the month).

        Args:
            current_date: Current trading date
            next_date: Next trading date (None if last date in dataset)

        Returns:
            True if month end, False otherwise
        """
        if next_date is None:
            return False  # Last day in dataset is NOT necessarily month-end

        return current_date.month != next_date.month

    def calculate_daily_unrealized_pnl(self, current_price: float) -> float:
        """
        Calculate mark-to-market unrealized P&L on all open positions.
        Inspired by sample strategy's df_result[c] = (df_price[c] * df_pos[c] - df_val[c])

        Args:
            current_price: Current raw_65_m1_price for mark-to-market

        Returns:
            Total unrealized P&L across all open positions
        """
        unrealized_pnl = 0.0

        for trade in self.open_positions:
            if trade.trade_type == "LONG":
                # LONG: profit when price rises above entry
                unrealized_pnl += (
                    current_price - trade.entry_price
                ) * trade.position_size
            else:  # SHORT
                # SHORT: profit when price falls below entry
                unrealized_pnl += (
                    trade.entry_price - current_price
                ) * trade.position_size

        return unrealized_pnl

    def calculate_next_day_unrealized_pnl(self, next_day_price: float) -> float:
        """
        Calculate unrealized P&L against next day's DSP (Daily Settlement Price).
        This is the correct method for proper daily P&L tracking.

        Args:
            next_day_price: Next day's DSP (Daily Settlement Price)

        Returns:
            Total unrealized P&L against next day's price
        """
        unrealized_pnl = 0.0

        for trade in self.open_positions:
            if trade.trade_type == "LONG":
                # LONG: profit when next day price rises above entry
                unrealized_pnl += (
                    next_day_price - trade.entry_price
                ) * trade.position_size
            else:  # SHORT
                # SHORT: profit when next day price falls below entry
                unrealized_pnl += (
                    trade.entry_price - next_day_price
                ) * trade.position_size

        return unrealized_pnl

    def get_position_list_string(self) -> str:
        """
        Get current open positions as a string representation.
        Format: [{entry_price: position_size}, {entry_price: position_size}]

        Returns:
            String representation of open positions
        """
        if not self.open_positions:
            return "[]"

        position_list = []
        for trade in self.open_positions:
            # Use negative size for SHORT positions
            size = (
                trade.position_size
                if trade.trade_type == "LONG"
                else -trade.position_size
            )
            position_list.append({trade.entry_price: size})

        return str(position_list)

    def get_current_position_summary(self) -> Dict[str, float]:
        """
        Get current position summary inspired by sample strategy position tracking.

        Returns:
            Dictionary with position metrics
        """
        long_positions = sum(
            trade.position_size
            for trade in self.open_positions
            if trade.trade_type == "LONG"
        )
        short_positions = sum(
            trade.position_size
            for trade in self.open_positions
            if trade.trade_type == "SHORT"
        )

        return {
            "long_positions": long_positions,
            "short_positions": short_positions,
            "net_position": long_positions - short_positions,  # Net exposure
            "gross_position": long_positions
            + short_positions,  # Total absolute exposure
            "num_open_trades": len(self.open_positions),
        }

    def log_daily_position_change(
        self,
        current_date: pd.Timestamp,
        current_price: float,
        realized_pnl: float,
        unrealized_pnl: float,
    ) -> None:
        """
        Log detailed daily position and P&L changes.

        Args:
            current_date: Current trading date
            current_price: Current raw_65_m1_price
            realized_pnl: Realized P&L from closed trades
            unrealized_pnl: Unrealized P&L from open positions
        """
        position_summary = self.get_current_position_summary()
        total_pnl = realized_pnl + unrealized_pnl

        # Log position changes
        if position_summary["num_open_trades"] > 0:
            logger.debug(
                f"{current_date.date()} | Price: ${current_price:.2f} | "
                f"Positions: {position_summary['net_position']:+.1f} net, "
                f"{position_summary['gross_position']:.1f} gross "
                f"({int(position_summary['long_positions'])}L, {int(position_summary['short_positions'])}S)"
            )

        # Log P&L breakdown if any activity
        if abs(realized_pnl) > 0.01 or abs(unrealized_pnl) > 0.01:
            logger.debug(
                f"{current_date.date()} | Daily P&L: Realized=${realized_pnl:.2f}, "
                f"Unrealized=${unrealized_pnl:.2f}, Total=${total_pnl:.2f}"
            )

    def close_all_positions(
        self, exit_date: pd.Timestamp, exit_price: float, exit_reason: str
    ) -> None:
        """
        Close all open positions.

        Args:
            exit_date: Date to close positions
            exit_price: Price to close at
            exit_reason: Reason for exit ("REVERSAL" or "EOM")
        """
        for trade in self.open_positions:
            trade.close_trade(exit_date, exit_price, exit_reason)
            logger.debug(
                f"Closed {trade.trade_type} position: "
                f"Entry ${trade.entry_price:.2f} → Exit ${exit_price:.2f}, "
                f"P&L: ${trade.pnl:.2f}, Reason: {exit_reason}"
            )

        # Clear open positions
        self.open_positions.clear()

    def open_position(
        self, entry_date: pd.Timestamp, entry_price: float, signal_type: str
    ) -> None:
        """
        Open new position.

        Args:
            entry_date: Entry date
            entry_price: Entry price
            signal_type: "LONG" or "SHORT"
        """
        trade = Trade(
            entry_date=entry_date,
            entry_price=entry_price,
            trade_type=signal_type,
            position_size=POSITION_SIZE,
        )

        self.trades.append(trade)
        self.open_positions.append(trade)

        logger.debug(
            f"Opened {signal_type} position: ${entry_price:.2f} on {entry_date.date()}"
        )

    def execute_backtest(self, test_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Execute complete backtesting on test data.

        Args:
            test_data: DataFrame with columns [date, raw_65_m1_price, predicted_ln_returns, actual_ln_returns]

        Returns:
            Dictionary with backtest results
        """
        logger.info("Executing LSTM trading strategy backtest...")
        logger.info(
            f"Test period: {test_data['date'].min()} to {test_data['date'].max()}"
        )
        logger.info(f"Total test samples: {len(test_data)}")

        # Reset strategy state including enhanced tracking
        self.trades.clear()
        self.open_positions.clear()
        self.daily_pnl.clear()
        self.equity_curve.clear()

        # Reset enhanced tracking lists
        self.daily_realized_pnl.clear()
        self.daily_unrealized_pnl.clear()
        self.daily_total_pnl.clear()
        self.daily_net_position.clear()
        self.daily_gross_position.clear()

        # Store test data for equity log generation
        self.test_data = test_data.copy()

        cumulative_realized_pnl = 0.0

        for row_idx in range(len(test_data)):
            row = test_data.iloc[row_idx]
            current_date = pd.Timestamp(row["date"])
            raw_price = row["raw_65_m1_price"]
            predicted_returns = row["predicted_ln_returns"]

            # Get next date for month-end detection
            next_date = None
            if row_idx < len(test_data) - 1:
                next_row = test_data.iloc[row_idx + 1]
                next_date = pd.Timestamp(next_row["date"])

            # Check for exits first (before new entries)

            # 1. Mandatory end-of-month exit (highest priority)
            if self.open_positions and self.is_month_end(current_date, next_date):
                self.close_all_positions(current_date, raw_price, "EOM")

            # 2. Direction reversal exit (if still have open positions)
            elif self.open_positions:
                # Check if any position should exit on direction reversal
                should_exit = any(
                    self.should_exit_on_reversal(trade, predicted_returns)
                    for trade in self.open_positions
                )

                if should_exit:
                    self.close_all_positions(current_date, raw_price, "REVERSAL")

            # Generate new entry signal (but NEVER on last trading date or month-end days)
            is_last_trading_date = next_date is None  # Last date in dataset
            is_month_end_day = self.is_month_end(current_date, next_date)

            # No new trades on last trading date or month-end days
            if not is_last_trading_date and not is_month_end_day:
                signal = self.generate_signal(raw_price, predicted_returns)
                if signal:
                    self.open_position(current_date, raw_price, signal)

            # Enhanced daily P&L calculation inspired by sample strategy
            # 1. Calculate realized P&L from closed trades (same as before)
            daily_realized_pnl = sum(
                trade.pnl
                for trade in self.trades
                if trade.exit_date == current_date and trade.pnl is not None
            )

            # 2. Calculate unrealized P&L on open positions (mark-to-market)
            daily_unrealized_pnl = self.calculate_daily_unrealized_pnl(raw_price)

            # 3. Calculate total P&L
            daily_total_pnl = daily_realized_pnl + daily_unrealized_pnl

            # 4. Track position sizes
            position_summary = self.get_current_position_summary()

            # 5. Update tracking lists
            self.daily_realized_pnl.append(daily_realized_pnl)
            self.daily_unrealized_pnl.append(daily_unrealized_pnl)
            self.daily_total_pnl.append(daily_total_pnl)
            self.daily_net_position.append(position_summary["net_position"])
            self.daily_gross_position.append(position_summary["gross_position"])

            # 6. Maintain backward compatibility
            self.daily_pnl.append(
                daily_realized_pnl
            )  # Keep old behavior for existing code
            cumulative_realized_pnl += daily_realized_pnl
            self.equity_curve.append(cumulative_realized_pnl)

            # 7. Granular logging
            self.log_daily_position_change(
                current_date, raw_price, daily_realized_pnl, daily_unrealized_pnl
            )

        # Close any remaining open positions at end of backtest
        if self.open_positions:
            final_date = pd.Timestamp(test_data.iloc[-1]["date"])
            final_price = test_data.iloc[-1]["raw_65_m1_price"]
            self.close_all_positions(final_date, final_price, "END_OF_DATA")

        logger.info(f"Backtest completed: {len(self.trades)} total trades executed")

        return self.calculate_performance_metrics(test_data)

    def calculate_performance_metrics(self, test_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate comprehensive performance metrics.

        Args:
            test_data: Original test data for context

        Returns:
            Dictionary with performance metrics
        """
        logger.info("Calculating trading performance metrics...")

        # Basic trade statistics
        closed_trades = [trade for trade in self.trades if not trade.is_open()]
        total_trades = len(closed_trades)

        if total_trades == 0:
            logger.warning("No closed trades found")
            return {"error": "No trades executed"}

        # P&L calculations
        trade_pnls = [trade.pnl for trade in closed_trades if trade.pnl is not None]
        total_pnl = sum(trade_pnls)
        winning_trades = [pnl for pnl in trade_pnls if pnl > 0]
        losing_trades = [pnl for pnl in trade_pnls if pnl <= 0]

        # Performance metrics
        win_rate = len(winning_trades) / total_trades * 100 if total_trades > 0 else 0
        avg_win = np.mean(winning_trades) if winning_trades else 0
        avg_loss = np.mean(losing_trades) if losing_trades else 0
        profit_factor = (
            sum(winning_trades) / abs(sum(losing_trades))
            if losing_trades
            else float("inf")
        )

        # Risk metrics - Corrected Sharpe ratio calculation using daily total P&L
        # Generate equity log to get correct daily P&L data
        equity_log_df = self.create_equity_log()

        if len(equity_log_df) > 1:
            # Use daily total P&L (realized + unrealized) for Sharpe calculation
            daily_total_pnl = equity_log_df["daily_total_pnl"].values

            # Calculate percentage returns based on cumulative equity
            daily_returns_pct = []
            cumulative_equity = 0.0  # Starting equity

            for pnl in daily_total_pnl:
                prev_equity = (
                    cumulative_equity if cumulative_equity != 0 else 1
                )  # avoid division by zero
                new_equity = cumulative_equity + pnl

                # Calculate percentage return
                daily_return_pct = (pnl / abs(prev_equity)) * 100
                daily_returns_pct.append(daily_return_pct)

                cumulative_equity = new_equity

            sharpe_ratio = (
                np.mean(daily_returns_pct) / np.std(daily_returns_pct)
                if np.std(daily_returns_pct) > 0
                else 0
            )
        else:
            daily_returns_pct = [0]
            sharpe_ratio = 0

        # Drawdown calculation
        peak = np.maximum.accumulate(self.equity_curve)
        drawdown = np.array(self.equity_curve) - peak
        max_drawdown = np.min(drawdown)
        max_drawdown_pct = (
            (max_drawdown / np.max(peak) * 100) if np.max(peak) > 0 else 0
        )

        # Holding periods
        holding_periods = [
            (trade.exit_date - trade.entry_date).days
            for trade in closed_trades
            if trade.exit_date
        ]
        avg_holding_period = np.mean(holding_periods) if holding_periods else 0

        # Trading activity analysis
        long_trades = [trade for trade in closed_trades if trade.trade_type == "LONG"]
        short_trades = [trade for trade in closed_trades if trade.trade_type == "SHORT"]

        results = {
            "backtest_summary": {
                "total_trades": total_trades,
                "long_trades": len(long_trades),
                "short_trades": len(short_trades),
                "test_period_days": len(test_data),
                "trades_per_month": (
                    float(total_trades) / (float(len(test_data)) / 30.44)
                    if len(test_data) > 0
                    else 0.0
                ),
            },
            "pnl_metrics": {
                "total_pnl": round(total_pnl, 2),
                "average_trade_pnl": (
                    round(total_pnl / total_trades, 2) if total_trades > 0 else 0
                ),
                "best_trade": round(max(trade_pnls), 2) if trade_pnls else 0,
                "worst_trade": round(min(trade_pnls), 2) if trade_pnls else 0,
                "final_equity": (
                    round(self.equity_curve[-1], 2) if self.equity_curve else 0
                ),
            },
            "win_loss_metrics": {
                "win_rate_pct": round(win_rate, 2),
                "profit_factor": (
                    round(profit_factor, 2)
                    if profit_factor != float("inf")
                    else "Infinite"
                ),
                "average_win": round(avg_win, 2),
                "average_loss": round(avg_loss, 2),
                "winning_trades": len(winning_trades),
                "losing_trades": len(losing_trades),
            },
            "risk_metrics": {
                "sharpe_ratio": round(sharpe_ratio, 3),
                "max_drawdown": round(max_drawdown, 2),
                "max_drawdown_pct": round(max_drawdown_pct, 2),
                "volatility_daily": (
                    round(np.std(daily_returns_pct), 4)
                    if len(daily_returns_pct) > 0
                    else 0
                ),
            },
            "time_metrics": {
                "average_holding_days": round(avg_holding_period, 1),
                "longest_trade_days": max(holding_periods) if holding_periods else 0,
                "shortest_trade_days": min(holding_periods) if holding_periods else 0,
            },
        }

        # Log key performance metrics
        logger.info("Trading Performance Summary:")
        logger.info(f"  Total P&L: ${total_pnl:.2f}")
        logger.info(
            f"  Total Trades: {total_trades} ({len(long_trades)} LONG, {len(short_trades)} SHORT)"
        )
        logger.info(
            f"  Win Rate: {win_rate:.1f}% ({len(winning_trades)}/{total_trades})"
        )
        logger.info(f"  Sharpe Ratio: {sharpe_ratio:.3f}")
        logger.info(f"  Max Drawdown: ${max_drawdown:.2f} ({max_drawdown_pct:.1f}%)")
        logger.info(f"  Average Holding: {avg_holding_period:.1f} days")

        return results

    def create_visualizations(self, save_dir: Path) -> Dict[str, str]:
        """
        Create trading strategy performance visualizations.

        Args:
            save_dir: Directory to save plots

        Returns:
            Dictionary mapping plot names to file paths
        """
        logger.info("Creating trading strategy visualizations...")

        save_dir.mkdir(parents=True, exist_ok=True)
        saved_plots = {}

        plt.style.use("seaborn-v0_8-whitegrid")
        palette = sns.color_palette("husl", 8)

        # 1. Equity curve
        _, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))

        # Equity curve
        ax1.plot(self.equity_curve, color=palette[0], linewidth=2)
        ax1.set_title("Trading Strategy Equity Curve", fontsize=14, fontweight="bold")
        ax1.set_ylabel("Cumulative P&L ($)")
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color="red", linestyle="--", alpha=0.7)

        # Daily P&L
        ax2.bar(
            range(len(self.daily_pnl)),
            self.daily_pnl,
            color=[palette[1] if pnl >= 0 else palette[2] for pnl in self.daily_pnl],
        )
        ax2.set_title("Daily P&L Distribution", fontsize=14, fontweight="bold")
        ax2.set_xlabel("Trading Day")
        ax2.set_ylabel("Daily P&L ($)")
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color="black", linestyle="-", alpha=0.5)

        plt.tight_layout()
        equity_plot_path = save_dir / "trading_equity_curve.png"
        plt.savefig(equity_plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        saved_plots["equity_curve"] = str(equity_plot_path)

        # 2. Trade analysis
        if self.trades:
            closed_trades = [trade for trade in self.trades if not trade.is_open()]

            if closed_trades:
                _, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

                # Trade P&L distribution
                pnls = [trade.pnl for trade in closed_trades if trade.pnl is not None]
                ax1.hist(pnls, bins=20, color=palette[0], alpha=0.7, edgecolor="black")
                ax1.set_title("Trade P&L Distribution", fontweight="bold")
                ax1.set_xlabel("P&L ($)")
                ax1.set_ylabel("Frequency")
                ax1.axvline(x=0, color="red", linestyle="--")

                # Holding periods
                holding_periods = [
                    (trade.exit_date - trade.entry_date).days
                    for trade in closed_trades
                    if trade.exit_date
                ]
                ax2.hist(
                    holding_periods,
                    bins=15,
                    color=palette[1],
                    alpha=0.7,
                    edgecolor="black",
                )
                ax2.set_title("Holding Period Distribution", fontweight="bold")
                ax2.set_xlabel("Days Held")
                ax2.set_ylabel("Frequency")

                # Monthly P&L
                monthly_pnl: Dict[pd.Period, float] = {}
                for trade in closed_trades:
                    if trade.exit_date and trade.pnl is not None:
                        month = trade.exit_date.to_period("M")
                        monthly_pnl[month] = monthly_pnl.get(month, 0.0) + trade.pnl

                if monthly_pnl:
                    months = list(monthly_pnl.keys())
                    pnls = list(monthly_pnl.values())
                    colors = [palette[1] if pnl >= 0 else palette[2] for pnl in pnls]
                    ax3.bar(range(len(pnls)), pnls, color=colors)
                    ax3.set_title("Monthly P&L", fontweight="bold")
                    ax3.set_xlabel("Month")
                    ax3.set_ylabel("Monthly P&L ($)")
                    ax3.set_xticks(range(len(months)))
                    ax3.set_xticklabels([str(m) for m in months], rotation=45)
                    ax3.axhline(y=0, color="black", linestyle="-", alpha=0.5)

                # Win/Loss by trade type
                long_pnls = [
                    trade.pnl
                    for trade in closed_trades
                    if trade.trade_type == "LONG" and trade.pnl is not None
                ]
                short_pnls = [
                    trade.pnl
                    for trade in closed_trades
                    if trade.trade_type == "SHORT" and trade.pnl is not None
                ]

                trade_types = []
                avg_pnls = []
                if long_pnls:
                    trade_types.append("LONG")
                    avg_pnls.append(np.mean(long_pnls))
                if short_pnls:
                    trade_types.append("SHORT")
                    avg_pnls.append(np.mean(short_pnls))

                if trade_types:
                    colors = [
                        palette[1] if pnl >= 0 else palette[2] for pnl in avg_pnls
                    ]
                    ax4.bar(trade_types, avg_pnls, color=colors)
                    ax4.set_title("Average P&L by Trade Type", fontweight="bold")
                    ax4.set_ylabel("Average P&L ($)")
                    ax4.axhline(y=0, color="black", linestyle="-", alpha=0.5)

                plt.tight_layout()
                analysis_plot_path = save_dir / "trading_analysis.png"
                plt.savefig(analysis_plot_path, dpi=300, bbox_inches="tight")
                plt.close()
                saved_plots["trade_analysis"] = str(analysis_plot_path)

        logger.info(f"Created {len(saved_plots)} trading visualization plots")
        return saved_plots

    def create_consolidated_trade_log(self) -> pd.DataFrame:
        """
        Create consolidated trade log with essential trade information only.
        P&L tracking is now handled in equity_log.csv.

        Returns:
            DataFrame with essential trade information
        """
        consolidated_data = []

        for trade in self.trades:
            # Only include essential trade information
            # P&L context columns removed as they're now tracked in equity_log.csv
            trade_dict = trade.to_dict()
            consolidated_data.append(trade_dict)

        return pd.DataFrame(consolidated_data)

    def create_equity_log(self) -> pd.DataFrame:
        """
        Create daily equity log with correct P&L tracking.

        This implements the correct P&L logic:
        1. Realized P&L: Only when trades actually close
        2. Unrealized P&L: Against next day's DSP (Daily Settlement Price)
        3. Cumulative Realized P&L: Running sum of realized P&L
        4. Position List: Open positions with entry prices and quantities

        Returns:
            DataFrame with daily equity tracking
        """
        if not hasattr(self, "test_data"):
            logger.warning("No test data available for equity log generation")
            return pd.DataFrame()

        logger.info("Creating daily equity log with correct P&L tracking...")

        equity_data = []
        cumulative_realized_pnl = 0.0

        # Process each day in chronological order
        for row_idx in range(len(self.test_data)):
            row = self.test_data.iloc[row_idx]
            current_date = pd.Timestamp(row["date"])
            current_price = row["raw_65_m1_price"]

            # Get next day's price for unrealized P&L calculation
            next_day_price = None
            if row_idx < len(self.test_data) - 1:
                next_row = self.test_data.iloc[row_idx + 1]
                next_day_price = next_row["raw_65_m1_price"]

            # Find which trades opened and closed on this date
            trades_opened_today = [
                trade
                for trade in self.trades
                if trade.entry_date.date() == current_date.date()
            ]

            trades_closed_today = [
                trade
                for trade in self.trades
                if trade.exit_date and trade.exit_date.date() == current_date.date()
            ]

            # Calculate realized P&L (only from trades that closed today)
            daily_realized_pnl = sum(
                trade.pnl for trade in trades_closed_today if trade.pnl is not None
            )

            # Update cumulative realized P&L
            cumulative_realized_pnl += daily_realized_pnl

            # Get open positions at end of day (after all entries/exits)
            open_positions_eod = []
            for trade in self.trades:
                # Position is open if it entered on or before today and exits after today (or never)
                entered_by_today = trade.entry_date.date() <= current_date.date()
                exits_after_today = (
                    trade.exit_date is None
                    or trade.exit_date.date() > current_date.date()
                )

                if entered_by_today and exits_after_today:
                    open_positions_eod.append(trade)

            # Calculate unrealized P&L against next day's DSP (if available)
            daily_unrealized_pnl = 0.0
            if next_day_price is not None and open_positions_eod:
                for trade in open_positions_eod:
                    if trade.trade_type == "LONG":
                        # LONG: profit when next day price rises above entry
                        daily_unrealized_pnl += (
                            next_day_price - trade.entry_price
                        ) * trade.position_size
                    else:  # SHORT
                        # SHORT: profit when next day price falls below entry
                        daily_unrealized_pnl += (
                            trade.entry_price - next_day_price
                        ) * trade.position_size

            # Create position list string (clean format without np.float64 wrapper)
            position_list = []
            for trade in open_positions_eod:
                # Use negative size for SHORT positions to match requested format
                size = (
                    trade.position_size
                    if trade.trade_type == "LONG"
                    else -trade.position_size
                )
                # Convert to regular Python float to avoid np.float64 in string representation
                entry_price = float(trade.entry_price)
                position_list.append({entry_price: size})

            position_list_str = str(position_list) if position_list else "[]"

            # Calculate daily total P&L
            daily_total_pnl = daily_realized_pnl + daily_unrealized_pnl

            # Add to equity data
            equity_data.append(
                {
                    "date": current_date,
                    "current_price": current_price,
                    "next_day_price": next_day_price,
                    "daily_realized_pnl": daily_realized_pnl,
                    "daily_unrealized_pnl": daily_unrealized_pnl,
                    "daily_total_pnl": daily_total_pnl,
                    "cumulative_realized_pnl": cumulative_realized_pnl,
                    "open_positions": position_list_str,
                    "num_open_positions": len(open_positions_eod),
                    "trades_opened_today": len(trades_opened_today),
                    "trades_closed_today": len(trades_closed_today),
                }
            )

        equity_df = pd.DataFrame(equity_data)

        logger.info(f"Generated equity log with {len(equity_df)} daily records")
        logger.info(f"Total realized P&L: ${cumulative_realized_pnl:.2f}")
        logger.info(
            f"Max open positions in single day: {equity_df['num_open_positions'].max()}"
        )

        return equity_df

    def export_results(
        self, performance_metrics: Dict[str, Any], save_dir: Path
    ) -> Dict[str, str]:
        """
        Export trading results to files with consolidated trade log and equity log.

        Args:
            performance_metrics: Performance metrics dictionary
            save_dir: Directory to save results

        Returns:
            Dictionary with file paths
        """
        logger.info("Exporting trading strategy results...")

        # Create trading results directory
        trading_dir = save_dir / "trading"
        trading_dir.mkdir(parents=True, exist_ok=True)

        # Export consolidated trade log with all context data (keep for backward compatibility)
        consolidated_trade_df = self.create_consolidated_trade_log()
        trade_log_path = trading_dir / "trade_log.csv"
        consolidated_trade_df.to_csv(trade_log_path, index=False)

        # Export NEW equity log with correct P&L tracking
        equity_log_df = self.create_equity_log()
        equity_log_path = trading_dir / "equity_log.csv"
        equity_log_df.to_csv(equity_log_path, index=False)

        # Export performance summary
        performance_summary = {
            "strategy_name": "LSTM Iron Ore Momentum Strategy",
            "buffer_threshold": BUFFER_THRESHOLD,
            "position_size": POSITION_SIZE,
            "backtest_timestamp": datetime.now().isoformat(),
            "performance_metrics": performance_metrics,
        }

        performance_path = trading_dir / "performance_summary.json"
        with open(performance_path, "w") as f:
            json.dump(performance_summary, f, indent=2, default=str)

        logger.info("Trading results exported:")
        logger.info(f"  Trade log (legacy): {trade_log_path}")
        logger.info(f"  ✅ NEW Equity log (correct P&L): {equity_log_path}")
        logger.info(f"  Performance summary: {performance_path}")
        logger.info(
            "  📊 Equity log contains correct daily P&L tracking with next-day DSP"
        )

        export_files = {
            "trade_log": str(trade_log_path),
            "equity_log": str(equity_log_path),  # NEW equity log
            "performance_summary": str(performance_path),
        }

        return export_files


def run_lstm_trading_strategy(
    test_results_path: Path, save_dir: Path = Path("results")
) -> Dict[str, Any]:
    """
    Factory function to run complete LSTM trading strategy.

    Args:
        test_results_path: Path to model_results.csv
        save_dir: Directory to save results

    Returns:
        Dictionary with strategy performance and file paths
    """
    logger.info("Starting LSTM trading strategy execution...")

    # Load test results
    test_data = pd.read_csv(test_results_path)
    test_data["date"] = pd.to_datetime(test_data["date"])

    # Initialize and run strategy
    strategy = LSTMTradingStrategy()
    performance_metrics = strategy.execute_backtest(test_data)

    # Create visualizations
    plots_dir = save_dir / "plots"
    plot_paths = strategy.create_visualizations(plots_dir)

    # Export results
    export_paths = strategy.export_results(performance_metrics, save_dir)

    # Combine results
    strategy_results = {
        "performance": performance_metrics,
        "export_files": export_paths,
        "plot_files": plot_paths,
        "strategy_config": {
            "buffer_threshold": BUFFER_THRESHOLD,
            "position_size": POSITION_SIZE,
        },
    }

    logger.info("LSTM trading strategy completed successfully")
    return strategy_results
