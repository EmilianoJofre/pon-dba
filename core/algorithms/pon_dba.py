"""
PON DBA Interface
Interfaces para algoritmos DBA modulares integradas de netPONPy
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from ..data.pon_request import Request
import numpy as np
from collections import deque


class DBAAlgorithmInterface(ABC):
    """Interfaz base para algoritmos DBA"""
    
    @abstractmethod
    def allocate_bandwidth(self, onu_requests: Dict[str, Any], 
                           total_bandwidth: float, **kwargs) -> Dict[str, float]:
        pass
    
    def select_next_request(self, available_requests: Dict[str, List[Request]], clock_time: float) -> Optional[Request]:
        for onu_id, request_list in available_requests.items():
            if request_list: return request_list[0]
        return None

    @abstractmethod
    def get_algorithm_name(self) -> str:
        pass

    def reset(self):
        pass
    
    def _flatten_requests(self, onu_requests: Dict[str, Any]) -> Dict[str, float]:
        """
        Método auxiliar ROBUSTO para convertir cualquier tipo de solicitud a float (MB).
        Maneja: Objetos Request, Diccionarios (HybridOLT) y Floats.
        """
        flat = {}
        for onu_id, req in onu_requests.items():
            # CASO 1: Objeto Request (Modo Orquestador/Ciclos)
            if hasattr(req, 'get_total_traffic'):
                flat[onu_id] = req.get_total_traffic()
            
            # CASO 2: Diccionario (Modo Eventos/Híbrido) <-- ESTO FALTABA
            elif isinstance(req, dict):
                # Sumar todos los valores del diccionario (bytes) y convertir a MB si es necesario
                # Asumimos que HybridOLT manda bytes o una unidad consistente.
                total_val = sum(v for v in req.values() if isinstance(v, (int, float)))
                # Si los valores vienen en bytes y el total_bandwidth está en MB,
                # aquí podrías necesitar dividir por (1024*1024). 
                # Por compatibilidad con FCFS legacy, asumimos que viene en la unidad correcta 
                # o que el algoritmo lo maneja. Si HybridOLT manda bytes:
                flat[onu_id] = float(total_val) 
            
            # CASO 3: Float/Int directo (Legacy)
            else:
                try:
                    flat[onu_id] = float(req)
                except (ValueError, TypeError):
                    flat[onu_id] = 0.0
        return flat


class FCFSDBAAlgorithm(DBAAlgorithmInterface):
    """FCFS Adaptado"""
    
    def allocate_bandwidth(self, onu_requests: Dict[str, Any], 
                           total_bandwidth: float, **kwargs) -> Dict[str, float]:
        
        # Usar método interno robusto
        flat_requests = self._flatten_requests(onu_requests)

        allocations = {}
        if not flat_requests: return allocations
            
        total_requested = sum(flat_requests.values())
        
        if total_requested <= total_bandwidth:
            allocations = flat_requests.copy()
        else:
            if total_requested > 0:
                for onu_id, requested in flat_requests.items():
                    proportion = requested / total_requested
                    allocations[onu_id] = total_bandwidth * proportion
            else:
                for onu_id in flat_requests:
                    allocations[onu_id] = 0.0
                
        return allocations

    def get_algorithm_name(self) -> str:
        return "FCFS"


class PriorityDBAAlgorithm(DBAAlgorithmInterface):
    """Algoritmo DBA basado en prioridades"""
    
    def __init__(self, starvation_threshold: float = 100.0):
        self.starvation_threshold = starvation_threshold / 1000.0
        self.traffic_priorities = {
            "highest": 1, "high": 2, "medium": 3, "low": 4, "lowest": 5
        }
    
    def allocate_bandwidth(self, onu_requests: Dict[str, Any], 
                          total_bandwidth: float, action: Any = None, **kwargs) -> Dict[str, float]:
        
        flat_requests = self._flatten_requests(onu_requests)
        allocations = {}
        
        if flat_requests:
            # Dar todo al primero que encuentre (lógica simple de prioridad)
            # Nota: Para prioridad real basada en TCONT, se necesitaría acceder al dict interno
            # pero para compatibilidad básica esto evita el crash.
            first_onu = next(iter(flat_requests))
            allocations[first_onu] = min(flat_requests[first_onu], total_bandwidth)
            
        return allocations
    
    def select_next_request(self, available_requests: Dict[str, List[Request]], 
                           clock_time: float) -> Optional[Request]:
        all_requests = []
        for onu_id, request_list in available_requests.items():
            for request in request_list:
                all_requests.append(request)
        
        if not all_requests: return None
        
        prioritized_requests = []
        for request in all_requests:
            waiting_time = clock_time - request.created_at
            if waiting_time >= self.starvation_threshold:
                prioritized_requests.append((0, request.created_at, request))
            else:
                request_priority = self._get_request_priority(request)
                prioritized_requests.append((request_priority, request.created_at, request))
        
        prioritized_requests.sort(key=lambda x: (x[0], x[1]))
        return prioritized_requests[0][2]
    
    def _get_request_priority(self, request: Request) -> int:
        if not request.traffic: return 999
        highest_priority = 999
        for traffic_type, amount in request.traffic.items():
            if amount and amount > 0:
                priority = self.traffic_priorities.get(traffic_type, 999)
                if priority < highest_priority:
                    highest_priority = priority
        return highest_priority
    
    def get_algorithm_name(self) -> str:
        return "Priority"


class RLDBAAlgorithm(DBAAlgorithmInterface):
    """RL-DBA Adaptado"""
    def allocate_bandwidth(self, onu_requests: Dict[str, Any], 
                           total_bandwidth: float, **kwargs) -> Dict[str, float]:
        
        flat_requests = self._flatten_requests(onu_requests)
        allocations = {}
        action = kwargs.get('action')
        
        if action is None:
            num_onus = len(flat_requests)
            equal_share = total_bandwidth / num_onus if num_onus > 0 else 0
            for onu_id in flat_requests:
                allocations[onu_id] = min(flat_requests[onu_id], equal_share)
        else:
            if hasattr(action, '__len__') and len(action) == len(flat_requests):
                action_sum = sum(action) if sum(action) > 0 else 1
                normalized_action = [a / action_sum for a in action]
                for i, onu_id in enumerate(sorted(flat_requests.keys())):
                    allocated_bandwidth = normalized_action[i] * total_bandwidth
                    allocations[onu_id] = min(flat_requests[onu_id], allocated_bandwidth)
            else:
                equal_share = total_bandwidth / len(flat_requests)
                for onu_id in flat_requests:
                    allocations[onu_id] = min(flat_requests[onu_id], equal_share)
        return allocations

    def get_algorithm_name(self) -> str:
        return "RL-DBA"


class StrictPriorityMinShareDBA(DBAAlgorithmInterface):
    """Algoritmo DBA con prioridad estricta y garantías mínimas."""
    
    def allocate_bandwidth(self, onu_requests: Dict[str, Any], total_bandwidth: float, action=None, **kwargs):
        
        # --- CAPA DE COMPATIBILIDAD MEJORADA ---
        converted_requests = {}
        for onu_id, req in onu_requests.items():
            if hasattr(req, 'traffic') and req.traffic:
                # Caso Request Object
                converted_requests[onu_id] = req.traffic
            elif isinstance(req, dict):
                # Caso HybridOLT (Diccionario directo de TCONTs)
                # Asumimos que ya viene en el formato {'high': bytes, ...}
                converted_requests[onu_id] = req
            elif hasattr(req, 'get_total_traffic'):
                val = req.get_total_traffic()
                converted_requests[onu_id] = self._distribute_dummy_traffic(val)
            else:
                try:
                    val = float(req)
                    converted_requests[onu_id] = self._distribute_dummy_traffic(val)
                except:
                    converted_requests[onu_id] = self._distribute_dummy_traffic(0.0)
        
        onu_requests = converted_requests
        # ------------------------------
        
        # Nota: HybridOLT suele enviar bytes. OLT envía MB (via Request object).
        # SP-MINSHARE trabaja internamente con bytes para precisión.
        # Si total_bandwidth viene en MB, lo pasamos a bytes.
        budget_bytes = int(total_bandwidth * 1024 * 1024)
        allocations = {}
        tcont_priorities = ['highest', 'high', 'medium', 'low', 'lowest']
        demand_per_tcont = {t: 0 for t in tcont_priorities}
        
        for onu_id, tdict in onu_requests.items():
            for tcont_id, req_val in tdict.items():
                # Detectar si el valor viene en MB (muy pequeño) o Bytes (muy grande)
                # Heurística simple: si es menor a 1000, probablemente es MB, si no Bytes.
                # O mejor: Estandarizar antes.
                # Para evitar romper lógica existente, asumimos que HybridOLT manda Bytes
                # y OLT manda MB.
                val_bytes = req_val
                if isinstance(req_val, float) and req_val < 10000: # Probablemente MB
                     val_bytes = int(req_val * 1024 * 1024)
                
                if tcont_id in demand_per_tcont:
                    demand_per_tcont[tcont_id] += max(0, int(val_bytes or 0))

        assigned = 0
        min_shares = {'highest': 0.25, 'high': 0.20, 'medium': 0.15, 'low': 0.10, 'lowest': 0.00}
        
        # 1. Asignar mínimos
        for tcont_id in tcont_priorities:
            min_bytes = int(min_shares.get(tcont_id, 0.0) * budget_bytes)
            if min_bytes <= 0 or demand_per_tcont[tcont_id] <= 0: continue
            
            total_dem_t = demand_per_tcont[tcont_id]
            for onu_id, tdict in onu_requests.items():
                raw_val = tdict.get(tcont_id, 0) or 0
                req = raw_val
                if isinstance(raw_val, float) and raw_val < 10000: req = int(raw_val * 1024 * 1024)

                req = max(0, int(req))
                if req <= 0: continue
                share = int(min_bytes * (req / total_dem_t))
                if share <= 0: continue
                give = min(share, req, budget_bytes - assigned)
                if give <= 0: continue
                allocations.setdefault(onu_id, {})[tcont_id] = allocations.get(onu_id, {}).get(tcont_id, 0) + give
                assigned += give
                if assigned >= budget_bytes: break
            if assigned >= budget_bytes: break

        # 2. Repartir sobrante
        if assigned < budget_bytes:
            for tcont_id in tcont_priorities:
                rem_total = 0
                rem_per_onu = {}
                for onu_id, tdict in onu_requests.items():
                    raw_val = tdict.get(tcont_id, 0) or 0
                    req = raw_val
                    if isinstance(raw_val, float) and raw_val < 10000: req = int(raw_val * 1024 * 1024)
                    req = max(0, int(req))
                    
                    already = allocations.get(onu_id, {}).get(tcont_id, 0)
                    rem = max(0, req - already)
                    if rem > 0:
                        rem_per_onu[onu_id] = rem
                        rem_total += rem
                if rem_total <= 0: continue

                leftover = budget_bytes - assigned
                if leftover <= 0: break

                for onu_id, rem in rem_per_onu.items():
                    share = int(leftover * (rem / rem_total))
                    if share <= 0: continue
                    give = min(share, rem, budget_bytes - assigned)
                    if give <= 0: continue
                    allocations.setdefault(onu_id, {})[tcont_id] = allocations.get(onu_id, {}).get(tcont_id, 0) + give
                    assigned += give
                    if assigned >= budget_bytes: break
                if assigned >= budget_bytes: break

        final_allocations = {}
        for onu_id, tdict in allocations.items():
            total_onu_bytes = sum(tdict.values())
            final_allocations[onu_id] = total_onu_bytes / (1024 * 1024)
        
        return final_allocations
    
    def _distribute_dummy_traffic(self, bandwidth):
        return {
            'highest': bandwidth * 0.4, 'high': bandwidth * 0.3,
            'medium': bandwidth * 0.2, 'low': bandwidth * 0.1, 'lowest': 0
        }
    
    def get_algorithm_name(self) -> str:
        return "SP-MINSHARE"


class JDFDBAAlgorithm(DBAAlgorithmInterface):
    """DF-DBA Jerárquico"""
    
    def __init__(self, buffer_size: int = 100, delta_t: float = 125e-6):
        self.N = buffer_size
        self.dt = delta_t
        self.surplus = 128 / (1024*1024)
        self.priorities = ['highest', 'high', 'medium', 'low', 'lowest']
        self.stats = {} 

    def _init_onu(self, onu_id):
        if onu_id not in self.stats:
            self.stats[onu_id] = {}
            for p in self.priorities:
                self.stats[onu_id][p] = {
                    'req': deque(maxlen=self.N), 'data': deque(maxlen=self.N)
                }
                for _ in range(2): 
                    self.stats[onu_id][p]['req'].append(0.0)
                    self.stats[onu_id][p]['data'].append(0.0)

    def _predict_tcont(self, onu_id, tcont):
        req_hist = np.array(self.stats[onu_id][tcont]['req'])
        data_hist = np.array(self.stats[onu_id][tcont]['data'])
        if len(req_hist) < 2: return 0.0
        
        req_prev = np.roll(req_hist, 1)
        req_prev[0] = req_hist[0]
        lambdas = (req_hist + data_hist - req_prev) / self.dt
        
        mu = np.mean(lambdas)
        sigma = np.std(lambdas)
        
        if sigma == 0: prediction_rate = mu
        else: prediction_rate = np.random.normal(mu, sigma)
        
        bytes_pred = (prediction_rate * self.dt) + self.surplus
        return max(0, bytes_pred)

    def allocate_bandwidth(self, onu_requests: Dict[str, Any], 
                           total_bandwidth: float, **kwargs) -> Dict[str, float]:
        
        last_transmitted_total = kwargs.get('last_transmitted', {})
        allocations = {oid: 0.0 for oid in onu_requests}
        bw_remaining = total_bandwidth
        predictions = {oid: {} for oid in onu_requests}
        
        for onu_id, req_obj in onu_requests.items():
            self._init_onu(onu_id)
            
            # --- MANEJO ROBUSTO DE ENTRADA ---
            current_traffic = {}
            total_req_size = 0.0
            
            if hasattr(req_obj, 'traffic'): # Request Object
                current_traffic = req_obj.traffic or {}
                total_req_size = req_obj.get_total_traffic()
            elif isinstance(req_obj, dict): # HybridOLT Dict
                # Convertir bytes a MB si parece ser bytes
                is_bytes = any(v > 1000 for v in req_obj.values())
                if is_bytes:
                    current_traffic = {k: v / (1024*1024) for k, v in req_obj.items()}
                else:
                    current_traffic = req_obj
                total_req_size = sum(current_traffic.values())
            else: # Float legacy
                try: total_req_size = float(req_obj)
                except: total_req_size = 0.0
            # ---------------------------------

            total_sent_size = last_transmitted_total.get(onu_id, 0.0)
            
            for p in self.priorities:
                curr_req = current_traffic.get(p, 0.0) or 0.0
                
                # Caso especial legacy sin detalle TCONT
                if total_req_size > 0 and not current_traffic and p == 'medium':
                     curr_req = total_req_size

                ratio = (curr_req / total_req_size) if total_req_size > 0 else 0
                curr_data = total_sent_size * ratio
                
                self.stats[onu_id][p]['req'].append(curr_req)
                self.stats[onu_id][p]['data'].append(curr_data)
                
                if curr_req == 0 and curr_data == 0:
                    predictions[onu_id][p] = 0.0
                else:
                    pred = self._predict_tcont(onu_id, p)
                    if pred < curr_req: predictions[onu_id][p] = curr_req
                    else: predictions[onu_id][p] = pred

        for p in self.priorities:
            if bw_remaining <= 0: break
            total_demand_p = sum(predictions[oid][p] for oid in onu_requests)
            if total_demand_p <= 0: continue
            
            if total_demand_p <= bw_remaining:
                for oid in onu_requests:
                    grant = predictions[oid][p]
                    allocations[oid] += grant
                    bw_remaining -= grant
            else:
                for oid in onu_requests:
                    req_p = predictions[oid][p]
                    share = (req_p / total_demand_p) * bw_remaining
                    allocations[oid] += share
                bw_remaining = 0

        return allocations

    def get_algorithm_name(self) -> str:
        return "J-DF-DBA"

    def reset(self):
        print("Reiniciando J-DF-DBA...")
        self.stats.clear()


class DFDBAAlgorithm(DBAAlgorithmInterface):
    """DF-DBA Clásico Adaptado"""
    
    def __init__(self, buffer_size: int = 100, frame_time_s: float = 125e-6):
        self.BUFFER_SIZE = buffer_size
        self.FRAME_TIME_S = frame_time_s
        self.SURPLUS_BYTES = 128
        self.onu_stats = {}

    def _initialize_onu_stats(self, onu_id: str):
        self.onu_stats[onu_id] = {
            "req_buffer": deque(maxlen=self.BUFFER_SIZE),
            "data_buffer": deque(maxlen=self.BUFFER_SIZE),
            "value_history": deque(maxlen=self.BUFFER_SIZE),
            "sum_x": 0.0, "sum_x_sq": 0.0, "mean": 0.0, "std_dev": 0.0
        }

    def _update_stats_incrementally(self, onu_id: str, new_value: float):
        stats = self.onu_stats[onu_id]
        if len(stats["value_history"]) == self.BUFFER_SIZE:
            old_value = stats["value_history"][0]
            stats["sum_x"] -= old_value
            stats["sum_x_sq"] -= old_value**2
            
        stats["sum_x"] += new_value
        stats["sum_x_sq"] += new_value**2
        stats["value_history"].append(new_value)
        
        n = len(stats["value_history"])
        if n > 0:
            stats["mean"] = stats["sum_x"] / n
            variance = (stats["sum_x_sq"] / n) - (stats["mean"]**2)
            stats["std_dev"] = np.sqrt(max(0, variance))

    def allocate_bandwidth(self, onu_requests: Dict[str, Any], 
                          total_bandwidth: float, **kwargs) -> Dict[str, float]:
        
        flat_requests = self._flatten_requests(onu_requests)
        active_onus = list(flat_requests.keys())
        if not active_onus: return {}

        last_transmitted = kwargs.get('last_transmitted', {})
        means = []; std_devs = []; fallback_predictions = {}

        for onu_id in active_onus:
            if onu_id not in self.onu_stats: self._initialize_onu_stats(onu_id)
            stats = self.onu_stats[onu_id]
            
            requested_bytes = flat_requests[onu_id]
            sent_bytes = last_transmitted.get(onu_id, 0.0)
            
            stats["req_buffer"].append(requested_bytes)
            stats["data_buffer"].append(sent_bytes)
            
            if len(stats["req_buffer"]) > 1:
                new_value = (stats["req_buffer"][-1] + stats["data_buffer"][-1] - stats["req_buffer"][-2])
                self._update_stats_incrementally(onu_id, new_value)

            if len(stats["value_history"]) < 2:
                fallback_predictions[onu_id] = requested_bytes + self.SURPLUS_BYTES
                means.append(0); std_devs.append(0)
            else:
                means.append(stats["mean"]); std_devs.append(stats["std_dev"])

        means_arr = np.array(means)
        std_devs_arr = np.array(std_devs)
        vectorized_predictions = np.random.normal(loc=means_arr, scale=std_devs_arr)

        allocations = {}
        for i, onu_id in enumerate(active_onus):
            if onu_id in fallback_predictions:
                predicted_bytes = fallback_predictions[onu_id]
            else:
                predicted_bytes = vectorized_predictions[i] + self.SURPLUS_BYTES
            allocations[onu_id] = max(0, predicted_bytes)
            
        return allocations

    def get_algorithm_name(self) -> str:
        return "DF-DBA"

    def reset(self):
        print("Reiniciando DF-DBA...")
        self.onu_stats.clear()