import os
import pickle
from typing import List, Literal

import pandas as pd

from consts import EXCEL_FILENAME_PH, TS_TEST_START, TS_END
from consts import ROUND_6, C_ENTRY_SIZE_MULTIPLIER_100
from consts import (
    C_DATE,
    C_LONG,
    C_SHORT,
    C_TOTAL,
)
from consts import EXIT_REVERSE, EXIT_EOM, EXIT_PRICING, EXIT_NON_PRICING
from consts import SimpleTrade, COLUMN_SIGNAL_DATE


CACHE_CONTRACT_DICT = {}


def get_65_by_contract() -> dict[pd.Period, pd.Series]:
    if CACHE_CONTRACT_DICT:
        return CACHE_CONTRACT_DICT

    price_dict = {}
    with open("Raw_M65F_DSP.pkl", "rb") as f:
        price_dict = pickle.load(f)
    for contract, ser in price_dict.items():
        ser = ser.ffill()
        mc_period = pd.Period(contract)
        CACHE_CONTRACT_DICT[mc_period] = ser

    return CACHE_CONTRACT_DICT


ExitOption = Literal[EXIT_PRICING, EXIT_NON_PRICING]
K_EXIT = "Exit"
K_PRICING_DATE_LIST = "Pricing Date List"
K_TC5_M2 = "TC5 M2"
K_M1_T = "M1(T)"
K_PREDICTED_M1_T1 = "Predicted M1(T+1)"


def strategy(
    exit_option: ExitOption, buffer: float, predict_file: str
) -> List[SimpleTrade]:
    assert exit_option == EXIT_NON_PRICING
    # Indicator
    df_ind = pd.read_excel(predict_file, index_col=0, parse_dates=True)
    settle_dict = get_65_by_contract()

    df_day = pd.read_excel(
        EXCEL_FILENAME_PH,
        sheet_name=0,
        index_col=0,
        parse_dates=True,
        usecols="A",
        names=[C_DATE],
    )
    df_day = df_day[
        (df_day.index >= TS_TEST_START) & (df_day.index <= TS_END)
    ].sort_index()
    monthly_last = set(df_day.groupby(df_day.index.to_period("M")).tail(5).index)

    # Logic
    trade_list = []
    for month, df_month in df_ind.groupby(df_ind.index.to_period("M")):
        m1 = month + 1
        m2 = month + 2
        if month > TS_END.to_period("M"):
            continue
        for ts, row in df_month.iterrows():
            if ts > TS_END:
                break
            cur_m1 = row[K_M1_T]
            predict_m1 = None
            if K_PREDICTED_M1_T1 in row:
                predict_m1 = row[K_PREDICTED_M1_T1]
            if pd.isna(predict_m1):
                continue
            signal_type, signal_logs = None, None
            predict_diff = round(predict_m1 - cur_m1, ROUND_6)
            # predict_diff = -predict_diff
            if predict_diff > buffer:
                signal_type = C_LONG
                signal_logs = (
                    f"{predict_m1:.2f} - {cur_m1} = {predict_diff:.2f} > {buffer}"
                )
            if predict_diff < -buffer:
                signal_type = C_SHORT
                signal_logs = (
                    f"{predict_m1:.2f} - {cur_m1} = {predict_diff:.2f} < -{buffer}"
                )
            # exit by back to 0
            for trade in trade_list:
                if trade.is_exit():
                    continue
                cur_entry_main = pd.Period(trade.entry_main)
                if cur_entry_main == m1:
                    exit_price = cur_m1
                elif cur_entry_main == m2:
                    exit_price = settle_dict[m2].loc[ts]
                    assert pd.notna(exit_price)
                else:
                    raise ValueError(f"{cur_entry_main}, {ts}")
                if trade.entry_type == C_LONG and predict_diff <= 0:
                    trade.exit_trade(
                        exit_ts=ts,
                        exit_price=exit_price,
                        exit_option=EXIT_REVERSE,
                        exit_signal_logs=f"{predict_m1:.2f} - {cur_m1} = {predict_diff:.2f} <= 0",
                    )
                elif trade.entry_type == C_SHORT and predict_diff >= 0:
                    trade.exit_trade(
                        exit_ts=ts,
                        exit_price=exit_price,
                        exit_option=EXIT_REVERSE,
                        exit_signal_logs=f"{predict_m1:.2f} - {cur_m1} = {predict_diff:.2f} >= 0",
                    )
            # entry
            if signal_type is None:
                continue
            current_position = (
                sum(
                    trade.entry_size
                    for trade in trade_list
                    if (not trade.is_exit() or trade.exit_ts > ts)
                )
                + 1
            )
            if ts in monthly_last:
                entry_main = f"{m2}"
                entry_price = settle_dict[m2].loc[ts]
                assert pd.notna(entry_price)
            else:
                entry_main = f"{m1}"
                entry_price = cur_m1
            trade = SimpleTrade(
                signal_ts=ts,
                entry_ts=ts,
                entry_type=signal_type,
                entry_size=1,
                entry_main=entry_main,
                entry_price=entry_price,
                entry_signal_logs=signal_logs,
                current_position=current_position,
            )
            trade_list.append(trade)
        # exit by EOM
        for trade in trade_list:
            if trade.is_exit():
                continue
            cur_entry_main = pd.Period(trade.entry_main)
            if cur_entry_main != m1:
                continue
            if exit_option == EXIT_NON_PRICING:
                ser = settle_dict[m1]
                ser = ser[ser.index.to_period("M") == month]
                trade.exit_trade(
                    exit_ts=ser.index[-1],
                    exit_price=ser.iloc[-1],
                    exit_option=EXIT_EOM,
                    exit_signal_logs=f"EOM on {ser.index[-1].date()}",
                )
            elif exit_option == EXIT_PRICING:
                ser = settle_dict[m1]
                if ser is None:
                    print(month)
                    print(settle_dict[m1])
                trade.foo.update({K_EXIT: EXIT_PRICING, K_PRICING_DATE_LIST: ser.index})
                trade.exit_trade(
                    exit_ts=ser.index[-1],
                    exit_price=ser.mean().round(ROUND_6),
                    exit_option=EXIT_EOM,
                    exit_signal_logs=f"EOM on {ser.index[-1].date()}",
                )
            else:
                assert ValueError(f"{exit_option} is not a valid exit_option")

    return trade_list


GS_FOLDER = "grid_search"
if not os.path.exists(GS_FOLDER):
    os.makedirs(GS_FOLDER)


def cal_mtm(
    key: str,
    trade_dto_list: List[SimpleTrade],
    index_list: List[pd.Timestamp],
    contract_dict: dict[pd.Period, pd.Series],
    is_save,
) -> tuple:
    action_list = []
    trade_list = []
    value_list = []
    for trade in trade_dto_list:
        trade_list.append(trade.to_dict())
        # entry
        entry_size = (
            trade.entry_size if trade.entry_type == C_LONG else -trade.entry_size
        )
        entry_size *= C_ENTRY_SIZE_MULTIPLIER_100
        action_list.append({C_DATE: trade.entry_ts, trade.entry_main: entry_size})
        value_list.append(
            {C_DATE: trade.entry_ts, trade.entry_main: entry_size * trade.entry_price}
        )
        # exit
        if trade.is_exit():
            action_list.append({C_DATE: trade.exit_ts, trade.entry_main: -entry_size})
            value_list.append(
                {
                    C_DATE: trade.exit_ts,
                    trade.entry_main: -entry_size * trade.exit_price,
                }
            )

    df_pos = pd.DataFrame(action_list).set_index(C_DATE).sort_index()
    df_pos = df_pos.groupby(df_pos.index).sum().reindex(index_list, fill_value=0)
    if is_save:
        df_pos.to_excel(f"{GS_FOLDER}/{key}_Action.xlsx")
    df_pos = df_pos.cumsum()

    df_val = pd.DataFrame(value_list).set_index(C_DATE).sort_index()
    df_val = df_val.groupby(df_val.index).sum().reindex(index_list, fill_value=0)
    df_val = df_val.cumsum()

    price_dict = {}
    for col in df_pos.columns:
        mc = pd.Period(col, "M")
        assert mc in contract_dict
        ser = contract_dict[mc]
        price_dict[col] = ser
    df_price = pd.DataFrame(price_dict).reindex(index_list, fill_value=0).sort_index()
    df_price = df_price.ffill()

    df_result = pd.DataFrame()
    for c in df_val.columns:
        df_result[c] = (df_price[c] * df_pos[c] - df_val[c]).round(6)
    df_result[C_TOTAL] = df_result.sum(axis=1)

    df_trade = pd.DataFrame(trade_list).set_index(COLUMN_SIGNAL_DATE).sort_index()
    if is_save:
        df_pos.to_excel(f"{GS_FOLDER}/{key}_Position.xlsx")
        df_val.to_excel(f"{GS_FOLDER}/{key}_Value.xlsx")
        df_price[df_pos.columns].to_excel(f"{GS_FOLDER}/{key}_DSP.xlsx")
        df_result.to_excel(f"{GS_FOLDER}/{key}_MTM.xlsx")
        df_trade.to_excel(f"{GS_FOLDER}/{key}_Trade.xlsx")

    return df_trade, df_result[C_TOTAL]
