# src/triage_feature.py
import numpy as np

class TriageAgent:
    def __init__(self, model, infer_engine):
        self.infer = infer_engine
        self.target = 'TYPE'
        # Get all nodes that are NOT the target (these are the potential questions)
        self.all_symptoms = [n for n in model.nodes() if n != self.target]

    def get_current_prediction(self, evidence):
        # Mengambil nilai prediksi saat ini berdasarkan evidence (gejala yang sudah dijawab)
        try:
            q = self.infer.query([self.target], evidence=evidence)
            probs = {state: val for state, val in zip(q.state_names[self.target], q.values)}
            # Sort by highest probability
            return dict(sorted(probs.items(), key=lambda item: item[1], reverse=True))
        except Exception as e:
            return {}

    def _calculate_entropy(self, probs):
        # Pakai Shannon Entropy
        # Metode untuk menghitung seberapa yakin kita terhadap prediksi saat ini berdasarkan semua probabilitas dari semua tipe
        entropy = 0
        for p in probs.values():
            if p > 0:
                entropy -= p * np.log2(p)
        return entropy

    def get_next_best_question(self, current_evidence):
        # Hitung probabilitas saat ini + kemudian hitung tingkat ketidakpastian (entropy)
        current_probs = self.get_current_prediction(current_evidence)
        current_entropy = self._calculate_entropy(current_probs)
        
        best_symptom = None
        max_info_gain = -1
        
        # Ambil gejala yang belum ditanyakan
        unanswered = [s for s in self.all_symptoms if s not in current_evidence]

        # Kalau semua sudah ditanyakan atau sudah dapat probabilitas tertinggi, stop
        top_prob = list(current_probs.values())[0] if current_probs else 0
        if not unanswered or top_prob > 0.95:
            return None

        # Looping untuk semua gejala yang belum ditanyakan
        for symptom in unanswered:
            # Cek berapa perkiraan probabilitas dari gejala itu, berdasarkan evidence (gejala yang sudah kita jawab) sejauh ini
            try:
                sym_prob_q = self.infer.query([symptom], evidence=current_evidence)
                p_yes = sym_prob_q.values[1] # Berapa probabilitasnya kalau yes
                p_no = sym_prob_q.values[0] # Berapa probabilitasnya kalau no
            except:
                continue

            # Kalau user menjawab yes, hitung apa pengaruhnya ke entropy (ketidakpastian) penyakit
            ev_yes = current_evidence.copy()
            ev_yes[symptom] = 1
            entropy_yes = self._calculate_entropy(self.get_current_prediction(ev_yes))

            # Kalau user menjawab no, hitung apa pengaruhnya ke entropy (ketidakpastian) penyakit
            ev_no = current_evidence.copy()
            ev_no[symptom] = 0
            entropy_no = self._calculate_entropy(self.get_current_prediction(ev_no))

            # Weighted Average (Expected Entropy) antara jawaban yes dan no
            expected_new_entropy = (p_yes * entropy_yes) + (p_no * entropy_no)

            # Information Gain = old entropy - new entropy
            # Lihat apakah dia mengurangi ketidakpastian kita terhadap diagnosis
            info_gain = current_entropy - expected_new_entropy

            # Pilih gejala dengan information gain tertinggi
            if info_gain > max_info_gain:
                max_info_gain = info_gain
                best_symptom = symptom

        return best_symptom