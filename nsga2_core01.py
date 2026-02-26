import datetime
import numpy as np
import random
import copy


class NSGA2Scheduler:
    def __init__(self, pop_size, n_gen, pc, pm, job_df, num_agvs):
        """
        初始化调度器
        :param job_df: 前端传入的 Pandas DataFrame (包含工件, 工序, 机器列表, 加工时间)
        """
        self.pop_size = pop_size
        self.n_gen = n_gen
        self.pc = pc
        self.pm = pm
        self.num_agvs = num_agvs

        # 1. 自动清洗并转换前端传入的 DataFrame
        self.job_data, self.job_ops = self._parse_frontend_data(job_df)

        # 2. 计算总工序数，用于确定染色体长度
        self.total_ops = sum(self.job_ops.values())

        # 3. 初始化最优种群记录，供后续提取甘特图数据使用
        self.best_population = []

    def _parse_frontend_data(self, df):
        """
        将前端 DataFrame 转换为解码器所需的结构化嵌套字典。
        处理前: DataFrame 中的字符串 "1, 2"
        处理后: job_data[job_id] = [{"machines": [1, 2], "times": [10.0, 12.0]}, ...]
        """
        job_data = {}
        job_ops = {}

        # 强制按“工件”和“工序”升序排序，防止用户在前端乱序录入导致逻辑错误
        df_sorted = df.sort_values(by=["工件", "工序"])

        for _, row in df_sorted.iterrows():
            job_id = int(row["工件"])

            # machine_str_list = str(row["机器列表"]).split(",")
            # time_str_list = str(row["加工时间"]).split(",")
            # 预处理：将中文逗号 "，" 统一替换为英文逗号 ","，防止用户输入全角符号导致报错
            machine_str = str(row["机器列表"]).replace("，", ",")
            time_str = str(row["加工时间"]).replace("，", ",")

            machine_str_list = machine_str.split(",")
            time_str_list = time_str.split(",")

            machines = [int(m.strip()) for m in machine_str_list]
            times = [float(t.strip()) for t in time_str_list]

            if job_id not in job_data:
                job_data[job_id] = []
                job_ops[job_id] = 0

            job_data[job_id].append({
                "machines": machines,
                "times": times
            })
            job_ops[job_id] += 1

        return job_data, job_ops

    # ================= 1. 种群初始化与编码 =================
    def _create_individual(self):
        """生成单个个体：包含 OS (工序顺序) 和 MS (机器选择) 编码"""
        os_list = []
        for job_id, op_count in self.job_ops.items():
            os_list.extend([job_id] * op_count)
        random.shuffle(os_list)

        ms_list = [random.random() for _ in range(self.total_ops)]

        return {
            "OS": os_list,
            "MS": ms_list,
            "f1": 0, "f2": 0,  # 目标值：时间, 能耗
            "rank": 0,  # 帕累托层级
            "distance": 0.0  # 拥挤度
        }

    def initialize_population(self):
        return [self._create_individual() for _ in range(self.pop_size)]

    # ================= 2. 非支配排序 =================
    def fast_non_dominated_sort(self, pop):
        """将种群划分为不同的帕累托前沿层级"""
        fronts = [[]]
        for p in pop:
            p["S"] = []
            p["n"] = 0

            for q in pop:
                if (p["f1"] <= q["f1"] and p["f2"] <= q["f2"]) and (p["f1"] < q["f1"] or p["f2"] < q["f2"]):
                    p["S"].append(q)
                elif (q["f1"] <= p["f1"] and q["f2"] <= p["f2"]) and (q["f1"] < p["f1"] or q["f2"] < p["f2"]):
                    p["n"] += 1

            if p["n"] == 0:
                p["rank"] = 0
                fronts[0].append(p)

        i = 0
        while len(fronts[i]) > 0:
            next_front = []
            for p in fronts[i]:
                for q in p["S"]:
                    q["n"] -= 1
                    if q["n"] == 0:
                        q["rank"] = i + 1
                        next_front.append(q)
            i += 1
            fronts.append(next_front)

        return fronts[:-1]

        # ================= 3. 拥挤度计算 =================

    def calculate_crowding_distance(self, front):
        """计算同一层级内个体的拥挤度，维持种群多样性"""
        l = len(front)
        if l == 0: return
        for p in front: p["distance"] = 0.0

        for obj in ["f1", "f2"]:
            front.sort(key=lambda x: x[obj])
            front[0]["distance"] = float('inf')
            front[-1]["distance"] = float('inf')

            f_max = front[-1][obj]
            f_min = front[0][obj]
            if f_max == f_min: continue

            for i in range(1, l - 1):
                front[i]["distance"] += (front[i + 1][obj] - front[i - 1][obj]) / (f_max - f_min)

    # ================= 4. 锦标赛选择 =================
    def tournament_selection(self, pop):
        """二元锦标赛选择：优先选择 rank 小的；rank 相同选择 distance 大的"""
        p1, p2 = random.sample(pop, 2)
        if p1["rank"] < p2["rank"]:
            return p1
        elif p1["rank"] > p2["rank"]:
            return p2
        else:
            return p1 if p1["distance"] > p2["distance"] else p2

    # ================= 5. 交叉与变异 (修复非法基因版) =================
    def crossover(self, p1, p2):
        """执行 POX (针对 OS 编码) 和 均匀交叉 (针对 MS 编码)"""
        if random.random() > self.pc:
            return copy.deepcopy(p1), copy.deepcopy(p2)

        c1, c2 = copy.deepcopy(p1), copy.deepcopy(p2)

        # --- 1. OS 编码的 POX (优先工序交叉) ---
        jobs = list(self.job_ops.keys())

        if len(jobs) > 1:
            num_in_set1 = random.randint(1, len(jobs) - 1)
            set1 = set(random.sample(jobs, num_in_set1))
        else:
            set1 = set(jobs)

        c1_os = [-1] * self.total_ops
        c2_os = [-1] * self.total_ops

        for i in range(self.total_ops):
            if p1["OS"][i] in set1:
                c1_os[i] = p1["OS"][i]
            if p2["OS"][i] in set1:
                c2_os[i] = p2["OS"][i]

        p2_leftovers = [x for x in p2["OS"] if x not in set1]
        p1_leftovers = [x for x in p1["OS"] if x not in set1]

        ptr1, ptr2 = 0, 0
        for i in range(self.total_ops):
            if c1_os[i] == -1:
                c1_os[i] = p2_leftovers[ptr1]
                ptr1 += 1
            if c2_os[i] == -1:
                c2_os[i] = p1_leftovers[ptr2]
                ptr2 += 1

        c1["OS"], c2["OS"] = c1_os, c2_os

        # --- 2. MS 编码的均匀交叉 ---
        for i in range(self.total_ops):
            if random.random() < 0.5:
                c1["MS"][i], c2["MS"][i] = c2["MS"][i], c1["MS"][i]

        return c1, c2

    def mutation(self, ind):
        """变异操作：OS 两点交换 + MS 重新随机选机"""
        if random.random() < self.pm:
            # --- 1. OS 交换变异 ---
            idx1, idx2 = random.sample(range(self.total_ops), 2)
            ind["OS"][idx1], ind["OS"][idx2] = ind["OS"][idx2], ind["OS"][idx1]

            # --- 2. MS 单点随机变异 ---
            ms_idx = random.randint(0, self.total_ops - 1)
            ind["MS"][ms_idx] = random.random()

        return ind

    # ================= 6. 核心解码逻辑 (推演时间轴) =================
    def _decode(self, ind):
        """
        核心推演引擎：将 OS 和 MS 转化为具体的机器与 AGV 排产时间表。
        """
        # 假设机器编号为 1 到 M，自动提取出所有提及的机器编号
        all_machines = set()
        for j_data in self.job_data.values():
            for op in j_data:
                all_machines.update(op["machines"])

        machine_times = {m: 0 for m in all_machines}
        job_times = {j: 0 for j in self.job_ops.keys()}
        job_locations = {j: 0 for j in self.job_ops.keys()}

        agv_times = {a: 0 for a in range(1, self.num_agvs + 1)}
        agv_locations = {a: 0 for a in range(1, self.num_agvs + 1)}

        op_counter = {j: 0 for j in self.job_ops.keys()}
        schedule = []
        total_agv_energy = 0

        for i, job_id in enumerate(ind["OS"]):
            op_idx = op_counter[job_id]
            op_counter[job_id] += 1

            op_data = self.job_data[job_id][op_idx]
            allowed_machines = op_data["machines"]
            proc_times = op_data["times"]

            ms_val = ind["MS"][i]
            m_idx = int(ms_val * len(allowed_machines))
            target_machine = int(allowed_machines[m_idx])
            proc_time = float(proc_times[m_idx])

            curr_loc = job_locations[job_id]
            if curr_loc != target_machine:
                best_agv = -1
                best_arrive_time = float('inf')
                empty_travel_time = 0

                for a in range(1, self.num_agvs + 1):
                    e_travel = abs(agv_locations[a] - curr_loc) * 2
                    arrive_at_job = max(agv_times[a] + e_travel, job_times[job_id])

                    if arrive_at_job < best_arrive_time:
                        best_arrive_time = arrive_at_job
                        best_agv = a
                        empty_travel_time = e_travel

                loaded_travel_time = abs(curr_loc - target_machine) * 2
                agv_start = best_arrive_time
                agv_end = agv_start + loaded_travel_time

                agv_times[best_agv] = agv_end
                agv_locations[best_agv] = target_machine
                job_times[job_id] = agv_end
                job_locations[job_id] = target_machine

                total_agv_energy += (empty_travel_time * 1 + loaded_travel_time * 2)

                schedule.append({
                    "task": f"AGV {best_agv}",
                    "resource": f"搬运 J{job_id} (M{curr_loc}->M{target_machine})",
                    "start": agv_start,
                    "end": agv_end
                })

            start_time = max(machine_times[target_machine], job_times[job_id])
            end_time = start_time + proc_time

            machine_times[target_machine] = end_time
            job_times[job_id] = end_time

            schedule.append({
                "task": f"机器 {target_machine}",
                "resource": f"J{job_id}-O{op_idx + 1}",
                "start": start_time,
                "end": end_time
            })

        makespan = max(machine_times.values()) if machine_times else 0
        return makespan, total_agv_energy, schedule

    # ================= 7. 适应度计算入口 =================
    def calculate_fitness(self, ind):
        """调用解码器，计算个体的 f1 和 f2"""
        makespan, energy, _ = self._decode(ind)
        ind["f1"] = makespan
        ind["f2"] = energy

    # ================= 8. 主循环与精英保留 =================
    def run(self, progress_callback):
        pop = self.initialize_population()
        for p in pop: self.calculate_fitness(p)

        history = []
        for g in range(self.n_gen):
            offspring = []
            while len(offspring) < self.pop_size:
                p1 = self.tournament_selection(pop)
                p2 = self.tournament_selection(pop)
                c1, c2 = self.crossover(p1, p2)
                offspring.extend([self.mutation(c1), self.mutation(c2)])

            for p in offspring: self.calculate_fitness(p)

            combined_pop = pop + offspring
            fronts = self.fast_non_dominated_sort(combined_pop)

            new_pop = []
            for front in fronts:
                self.calculate_crowding_distance(front)
                if len(new_pop) + len(front) <= self.pop_size:
                    new_pop.extend(front)
                else:
                    front.sort(key=lambda x: x["distance"], reverse=True)
                    new_pop.extend(front[:self.pop_size - len(new_pop)])
                    break

            pop = new_pop

            best_f1 = min(p["f1"] for p in pop)
            best_f2 = min(p["f2"] for p in pop)
            history.append([best_f1, best_f2])
            progress_callback((g + 1) / self.n_gen)

        self.best_population = pop
        return history

    # ================= 9. 输出给前端甘特图的数据 =================
    def get_best_schedule_data(self):
        """提取帕累托前沿中最优解的排产明细"""
        if not hasattr(self, 'best_population') or not self.best_population:
            return []

        best_ind = min(self.best_population, key=lambda x: x["f1"])
        _, _, schedule = self._decode(best_ind)
        return schedule


def parse_schedule_to_gantt(raw_schedule_data, start_hour=8, time_unit="minutes"):
    """将算法输出的时间，转化为 Plotly 甘特图所需的 datetime 格式"""
    base_time = datetime.datetime.today().replace(hour=start_hour, minute=0, second=0, microsecond=0)
    gantt_list = []
    for item in raw_schedule_data:
        if time_unit == "minutes":
            start_dt = base_time + datetime.timedelta(minutes=float(item["start"]))
            end_dt = base_time + datetime.timedelta(minutes=float(item["end"]))
        else:
            start_dt = base_time + datetime.timedelta(seconds=float(item["start"]))
            end_dt = base_time + datetime.timedelta(seconds=float(item["end"]))

        gantt_list.append({
            "Task": item["task"],
            "Start": start_dt,
            "Finish": end_dt,
            "Resource": item["resource"]
        })

    return gantt_list