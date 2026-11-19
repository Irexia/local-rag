import json, re, hashlib, sys
from collections import Counter
from pathlib import Path

DATA_DIR = Path("./data_sample")
OUT = Path("./testset.jsonl")
HELDOUT_FRACTION = 0.30
SPLIT_SALT = "rag-eval-v1"

S, B, I, Z, C = ("solar_panels.txt", "battery_storage.txt", "inverter_guide.txt",
                 "system_sizing.txt", "commissioning_and_monitoring.txt")

UNANS = "Not covered by the documents. The system should say it does not know."
rows = []


def Q(cat, question, answer, *spans):
    """spans: (source_file, verbatim_text) tuples"""
    rows.append({"category": cat, "question": question, "ground_truth": answer,
                 "gold": [{"source": s, "text": t} for s, t in spans]})


# =====================================================================================
# A. ORIGINAL 17 QUESTIONS (kept as-is, now with span labels)
# =====================================================================================
Q("single_fact", "What is the optimal tilt angle for a site at 40 degrees north latitude?",
  "The optimal tilt angle equals the site's latitude, so a site at 40 degrees north should have panels tilted at 40 degrees from horizontal.",
  (S, "The optimal tilt angle equals the latitude of the installation site. For a site at 40 degrees north latitude, the panel should be tilted at 40 degrees from horizontal."))
Q("multi_fact", "How often should solar panels be cleaned and what should be used?",
  "Panels should be cleaned quarterly with a soft brush and deionised water. Abrasive cleaners must not be used.",
  (S, "Quarterly: Clean panels with a soft brush and deionised water. Never use abrasive cleaners or pressure washers above 40 bar, as these can crack the tempered glass cover."))
Q("single_fact", "What is the maximum pressure washer pressure allowed when cleaning solar panels?",
  "Pressure washers must not be used above 40 bar, because higher pressure can crack the tempered glass cover.",
  (S, "Never use abrasive cleaners or pressure washers above 40 bar, as these can crack the tempered glass cover."))
Q("multi_fact", "How does shading affect solar panel output and how should it be assessed?",
  "Even partial shading on a single cell can reduce the output of an entire string by up to 80%. A shading study using a solace sensor or PV*SOL software must be done at 9 AM, noon and 3 PM across equinox and solstice dates.",
  (S, "Even partial shading on a single cell can reduce the output of an entire string by up to 80%."),
  (S, "A shading study using a solace sensor or PV*SOL software must be performed at three times of day (9 AM, noon, 3 PM) across equinox and solstice dates."))
Q("multi_fact", "What roof load capacity is required for residential solar installations?",
  "Residential roofs must support a dead load of at least 30 kg per square meter for the mounting hardware and panels, plus a live load of 1.5 kN per square meter in snow regions. A licensed structural engineer must sign off before mounting.",
  (S, "Residential roofs must support a dead load of at least 30 kg per square meter for the mounting hardware and panels, plus a live load of 1.5 kN per square meter for snow regions."),
  (S, "A licensed structural engineer must sign off before mounting."))
Q("multi_fact", "How much energy should a 6 kW system produce per day at 4.2 peak sun hours in a moderate climate?",
  "Before losses it generates 25.2 kWh per day (6,000 W x 4.2 h). After 12-20% system losses, a 6 kW system in a moderate climate should realistically produce 20 to 22 kWh per day.",
  (S, "For a 6 kW system at a site with 4.2 peak sun hours per day, the expected daily generation is 6,000 W x 4.2 h = 25.2 kWh per day, before system losses."),
  (S, "Total expected system loss: 12-20%"),
  (S, "Therefore a 6 kW system in a moderate climate should realistically produce between 20 and 22 kWh per day."))
Q("single_fact", "How much does panel output drop with temperature?",
  "Temperature derating is approximately 0.35% per degree Celsius above 25 degrees ambient.",
  (S, "Temperature derating: approximately 0.35% per degree Celsius above 25 degrees ambient"))
Q("multi_fact", "What is the minimum DC cable cross-section for runs under 20 metres, and what are the inverter output cable requirements?",
  "DC cabling for runs under 20 metres must be at least 6 mm2. The inverter output cable must be at least 10 mm2 and fused at 80% of the inverter rated current.",
  (S, "Run DC cabling (minimum 6 mm² cross-section for runs under 20 metres) through a waterproof conduit from the panel array to the inverter location."),
  (S, "The inverter output cable must be minimum 10 mm² and fused at 80% of the inverter rated current."))
Q("multi_fact", "What does fault code F-003 mean and what should be done?",
  "F-003 means inverter over-temperature. Ensure ventilation clearance of at least 15 cm on all sides.",
  (S, "Code F-003: Inverter over-temperature. Ensure ventilation clearance of at least 15 cm on all sides."))
Q("single_fact", "What temperature range can lithium iron phosphate batteries operate in?",
  "LiFePO4 batteries operate from 0 to 45 degrees Celsius.",
  (B, "Operating temperature: 0 to 45 degrees C"))
Q("single_fact", "Which battery type has the lowest initial cost but the highest lifetime cost, and why?",
  "Lead-acid (AGM/Gel) has the lowest initial cost but the highest lifetime cost because of the need for replacement.",
  (B, "Cost: Lowest initial, highest lifetime (due to replacement)"))
Q("multi_fact", "How does the energy density of NMC batteries compare to other chemistries?",
  "NMC has an energy density of 150-220 Wh/kg, higher than LiFePO4 (90-160 Wh/kg) and lead-acid (30-40 Wh/kg).",
  (B, "Energy density: 90-160 Wh/kg"), (B, "Energy density: 150-220 Wh/kg"), (B, "Energy density: 30-40 Wh/kg"))
Q("multi_fact", "What does the BMS monitor and control in a lithium battery system?",
  "The BMS monitors and controls cell voltage (balancing cells within 50 mV), temperature (charging halts above 45 C or below 0 C), charge and discharge current limits, and state of charge estimation.",
  (B, "Individual cell balancing to keep all cells within 50 mV of each other."),
  (B, "Charging must be halted if any cell exceeds 45 degrees C or drops below 0 degrees C."),
  (B, "Both charge and discharge current must be limited to the cell datasheet maximum."),
  (B, "Uses a combination of voltage lookup tables, current integration (Coulomb counting), and periodic open-circuit voltage calibration."))
Q("single_fact", "What is the cycle life of LiFePO4 cells at 60% depth of discharge?",
  "LiFePO4 cells deliver 3,000+ cycles at 60% depth of discharge.",
  (B, "60% DOD: 3,000+ cycles"))
Q("multi_fact", "How much energy per day does a 10 kWh battery deliver at 60% DOD and how long will it last?",
  "It delivers about 6 kWh per day and lasts around 3,000 days (about 8 years) before capacity drops below 80% of rated.",
  (B, "A 10 kWh system used daily at 60% DOD will deliver approximately 6 kWh per day and last around 3,000 days (8 years) before capacity drops below 80% of rated."))
Q("multi_fact", "What safety requirements apply to an outdoor residential battery installation?",
  "The enclosure must be IP65 or higher, installed on a non-combustible surface, with a Class D fire extinguisher within 3 metres and a clearly labelled DC disconnect within 2 metres. Wiring must be rated for max continuous current plus 25%.",
  (B, "The battery enclosure must be IP65 rated or higher for outdoor installation."),
  (B, "The battery must be installed on a non-combustible surface."),
  (B, "A Class D fire extinguisher must be available within 3 metres."),
  (B, "The DC disconnect switch must be clearly labelled and located within 2 metres of the battery."),
  (B, "All wiring must be rated for the maximum continuous current plus 25% for fault conditions."))
Q("multi_fact", "How is a typical 10 kWh LiFePO4 system configured?",
  "It uses 16 LiFePO4 cells (3.2 V nominal, 100 Ah each, matched within 1%) in series for 51.2 V, with a 16S BMS rated 100 A continuous and 200 A peak, and a 5 kW hybrid inverter/charger.",
  (B, "16 x LiFePO4 cells (3.2V nominal each) in series = 51.2V"),
  (B, "BMS: 16S, 100A continuous, 200A peak"),
  (B, "Cells: 100 Ah each, matched within 1% capacity"),
  (B, "Inverter/Charger: 5 kW hybrid, 51.2V input, 230V AC output"))

# =====================================================================================
# B. SINGLE-FACT (one fact, one passage)
# =====================================================================================
# --- solar_panels.txt
Q("single_fact", "How much more energy do panels produce when facing true south in the northern hemisphere?",
  "Up to 25% more energy.",
  (S, "Panels produce up to 25% more energy when facing true south (in the northern hemisphere) or true north (southern hemisphere)."))
Q("single_fact", "How much annual yield do east-west facing panels lose?",
  "East-west orientations lose approximately 10-15% in annual yield.",
  (S, "East-west orientations lose approximately 10-15% in annual yield."))
Q("single_fact", "By how much can a seasonal tilt adjustment increase annual output?",
  "A seasonal adjustment of plus or minus 15 degrees can increase annual output by 3-5%.",
  (S, "A seasonal adjustment of plus or minus 15 degrees can increase annual output by 3-5%."))
Q("single_fact", "Who has to sign off on the roof before panels are mounted?",
  "A licensed structural engineer.",
  (S, "A licensed structural engineer must sign off before mounting."))
Q("single_fact", "What live load must a roof support in snow regions?",
  "A live load of 1.5 kN per square meter.",
  (S, "plus a live load of 1.5 kN per square meter for snow regions."))
Q("single_fact", "What unit are solar panels rated in?",
  "Watts-peak (Wp).",
  (S, "Panels are rated in watts-peak (Wp)."))
Q("single_fact", "What is the typical size range of a residential solar system?",
  "A typical residential system ranges from 3 kW to 10 kW.",
  (S, "A typical residential system ranges from 3 kW to 10 kW."))
Q("single_fact", "What is the inverter efficiency loss range used for system loss calculations?",
  "Inverter efficiency loss is 5-8%.",
  (S, "Inverter efficiency loss: 5-8%"))
Q("single_fact", "What soiling losses should be expected in arid regions?",
  "Soiling losses are 3-5% in arid regions (1-2% in moderate climates).",
  (S, "Soiling losses (dust, bird droppings): 3-5% in arid regions, 1-2% in moderate climates"))
Q("single_fact", "What is the total expected system loss for a PV system?",
  "Total expected system loss is 12-20%.",
  (S, "Total expected system loss: 12-20%"))
Q("single_fact", "What mismatch loss range should be allowed for?",
  "Mismatch losses are 1-3%.",
  (S, "Mismatch losses: 1-3%"))
Q("single_fact", "Which sealant is used for the pre-drilled roof holes when mounting the racking?",
  "Sikaflex 291 or an equivalent construction adhesive.",
  (S, "Mount the racking system to the roof structure using pre-drilled holes sealed with Sikaflex 291 or equivalent construction adhesive and 16g x 100mm screws."))
Q("single_fact", "What can a reverse polarity connection damage?",
  "A reverse polarity connection can damage the inverter.",
  (S, "A reverse polarity connection can damage the inverter."))
Q("single_fact", "What cable is required to ground the panel frames and inverter chassis?",
  "Grounding wire must be minimum 4 mm2 green/yellow insulated cable.",
  (S, "Grounding wire must be minimum 4 mm² green/yellow insulated cable."))
Q("single_fact", "How often should MC4 connectors be replaced and why?",
  "Every 5 years as a precaution, because UV degradation of the polymer housing can lead to arcing.",
  (S, "Every 5 years: Replace all DC connectors (MC4) as a precaution, as UV degradation of the polymer housing can lead to arcing."))
Q("single_fact", "What does fault code F-001 mean?",
  "Grid voltage out of range. Check the utility connection.",
  (S, "Code F-001: Grid voltage out of range. Check utility connection."))
Q("single_fact", "What does fault code F-004 indicate and what should be inspected?",
  "A ground fault was detected. Inspect the DC cabling for insulation damage.",
  (S, "Code F-004: Ground fault detected. Inspect DC cabling for insulation damage."))
Q("single_fact", "What should be checked when fault code F-005 appears?",
  "F-005 is a communication failure between inverter and monitoring gateway. Check the RS485 or Wi-Fi link.",
  (S, "Code F-005: Communication failure between inverter and monitoring gateway. Check RS485 or Wi-Fi link."))
Q("single_fact", "What does the monthly solar maintenance inspection cover?",
  "A visual inspection for physical damage, loose mounting hardware, and abnormal inverter error codes.",
  (S, "Monthly: Visual inspection for physical damage, loose mounting hardware, and abnormal inverter error codes."))

# --- battery_storage.txt
Q("single_fact", "How many cycles does NMC deliver at 80% depth of discharge?",
  "NMC delivers 1,000-2,000 cycles at 80% depth of discharge.",
  (B, "Cycle life: 1,000-2,000 cycles at 80% depth of discharge"))
Q("single_fact", "What is the cycle life of lead-acid batteries at 50% depth of discharge?",
  "300-500 cycles.",
  (B, "Cycle life: 300-500 cycles at 50% depth of discharge"))
Q("single_fact", "What temperature range can lead-acid batteries operate in?",
  "-20 to 50 degrees C.",
  (B, "Operating temperature: -20 to 50 degrees C"))
Q("single_fact", "Which chemistry is best for EVs and shorter-duration storage?",
  "NMC (Nickel Manganese Cobalt).",
  (B, "Best for: EVs, shorter-duration storage"))
Q("single_fact", "What is the thermal stability of LiFePO4?",
  "Excellent, with a non-flammable electrolyte.",
  (B, "Thermal stability: Excellent, non-flammable electrolyte"))
Q("single_fact", "How many cycles can LiFePO4 deliver at 80% depth of discharge?",
  "2,000-3,000 cycles at 80% depth of discharge.",
  (B, "Cycle life: 2,000-3,000 cycles at 80% depth of discharge"))
Q("single_fact", "What cell voltage drift indicates an aging or defective cell?",
  "A drift beyond 200 mV.",
  (B, "A drift beyond 200 mV indicates an aging or defective cell."))
Q("single_fact", "What sensors measure cell temperature in a BMS?",
  "NTC thermistors placed on each cell.",
  (B, "NTC thermistors placed on each cell."))
Q("single_fact", "What continuous discharge current is a typical 100 Ah LiFePO4 cell rated for?",
  "100 A continuous discharge (1C).",
  (B, "A typical 100 Ah LiFePO4 cell is rated for 100A continuous discharge (1C)."))
Q("single_fact", "How does a BMS estimate state of charge?",
  "A combination of voltage lookup tables, current integration (Coulomb counting), and periodic open-circuit voltage calibration.",
  (B, "Uses a combination of voltage lookup tables, current integration (Coulomb counting), and periodic open-circuit voltage calibration."))
Q("single_fact", "How many cycles does LiFePO4 last at 20% depth of discharge?",
  "8,000+ cycles.",
  (B, "20% DOD: 8,000+ cycles"))
Q("single_fact", "What is the LiFePO4 cycle life at 100% depth of discharge and is it recommended?",
  "800-1,200 cycles, and it is not recommended.",
  (B, "100% DOD: 800-1,200 cycles (not recommended)"))
Q("single_fact", "How much ventilation airflow is needed per kWh for indoor battery installations?",
  "At least 0.5 m3/min of airflow per kWh of installed capacity.",
  (B, "At least 0.5 m³/min of airflow per kWh of installed capacity for indoor installations."))
Q("single_fact", "Which standards must a battery installation comply with?",
  "Local electrical codes such as IEC 62619, UL 1973, or equivalent.",
  (B, "The installation must comply with local electrical codes (IEC 62619, UL 1973, or equivalent)."))
Q("single_fact", "What usable capacity range do typical residential battery systems have?",
  "5 kWh to 20 kWh of usable capacity.",
  (B, "Typical residential systems range from 5 kWh to 20 kWh of usable capacity."))
Q("single_fact", "What is the expected lifetime of the 10 kWh system at 50% daily depth of discharge?",
  "Approximately 4,000 cycles, or 10+ years.",
  (B, "Expected lifetime at 50% daily DOD: approximately 4,000 cycles, or 10+ years"))
Q("single_fact", "What is the round-trip efficiency of LiFePO4 batteries?",
  "92-96%.",
  (B, "Round-trip efficiency: LiFePO4 92-96%, NMC 90-95%, lead-acid 75-85%"))
Q("single_fact", "What is the self-discharge rate of lead-acid batteries?",
  "3-5% per month.",
  (B, "Self-discharge: LiFePO4 2-3% per month, lead-acid 3-5% per month"))

# --- inverter_guide.txt
Q("single_fact", "What is the MPPT voltage window of a typical 5 kW residential hybrid inverter?",
  "120 V to 500 V.",
  (I, "MPPT voltage window: 120 V to 500 V"))
Q("single_fact", "What is the maximum DC input voltage of a typical 5 kW residential hybrid inverter?",
  "600 V.",
  (I, "Maximum DC input voltage: 600 V"))
Q("single_fact", "How much does open-circuit voltage rise as panels get colder?",
  "Approximately 0.3% per degree Celsius below 25 degrees.",
  (I, "Open-circuit voltage (Voc) rises as panels get colder, by approximately 0.3% per degree Celsius below 25 degrees."))
Q("single_fact", "What DC/AC ratio is recommended for an inverter?",
  "A ratio between 1.1 and 1.3.",
  (I, "A ratio between 1.1 and 1.3 is recommended."))
Q("single_fact", "What is the surge capacity of a 5 kW class hybrid inverter?",
  "7.5 kW for 10 seconds.",
  (I, "Surge capacity: 7.5 kW for 10 seconds"))
Q("single_fact", "What is the backup changeover time of the hybrid inverter?",
  "Under 20 ms.",
  (I, "Backup changeover time: under 20 ms"))
Q("single_fact", "What does fault code F-006 mean?",
  "Battery communication lost. Check the CAN bus cable between the battery BMS and the inverter.",
  (I, "Code F-006: Battery communication lost. Check the CAN bus cable between the battery BMS and the inverter."))
Q("single_fact", "At what pack voltage does fault code F-007 occur?",
  "F-007 is battery under-voltage: pack voltage below 44.8 V.",
  (I, "Code F-007: Battery under-voltage. Pack voltage is below 44.8 V."))
Q("single_fact", "How often should inverter cooling fans be replaced?",
  "Inspect annually and replace every 7 years.",
  (I, "Cooling fans: inspect annually and replace every 7 years"))
Q("single_fact", "What is the design life of an inverter?",
  "10-15 years.",
  (I, "Inverter design life: 10-15 years"))
Q("single_fact", "What is the weighted efficiency of residential inverters?",
  "92-95%.",
  (I, "Weighted (European) efficiency is 92-95% for residential units"))

# --- system_sizing.txt
Q("single_fact", "What share of daily consumption is typically used in the evening?",
  "35-45% of daily consumption.",
  (Z, "Evening load is the energy used between 6 PM and 11 PM, when solar generation is zero. It is typically 35-45% of daily consumption."))
Q("single_fact", "How many peak sun hours do the Mediterranean and US Southwest get?",
  "5.0-6.5 hours per day.",
  (Z, "Mediterranean and US Southwest: 5.0-6.5 hours per day"))
Q("single_fact", "How much electricity does an average home use per day?",
  "10-15 kWh per day.",
  (Z, "Average home: 10-15 kWh per day"))
Q("single_fact", "How many panels and how much roof area does a 6 kW array need?",
  "15 panels and about 28.5 square metres of roof area.",
  (Z, "A 6 kW array therefore needs 15 panels and about 28.5 square metres of roof area."))
Q("single_fact", "How fast do solar panels degrade each year?",
  "About 0.5% per year, leaving about 87% of original output after 25 years.",
  (Z, "Panel output degrades by about 0.5% per year. After 25 years expect about 87% of the original output."))
Q("single_fact", "What formula gives the nominal battery capacity from the usable capacity?",
  "Nominal capacity = usable capacity / planned depth of discharge.",
  (Z, "Nominal capacity = usable capacity / planned depth of discharge."))
Q("single_fact", "How much energy do typical critical loads consume during a 12-hour outage?",
  "The four typical critical loads total 330 W and consume 3.96 kWh over 12 hours.",
  (Z, "These four loads total 330 W. Over a 12-hour outage they consume 3.96 kWh."))

# --- commissioning_and_monitoring.txt
Q("single_fact", "What voltage and minimum result apply to the insulation resistance test?",
  "Test at 500 V DC; the result must be at least 1 MOhm between each live conductor and earth.",
  (C, "Insulation resistance: Test at 500 V DC. The result must be at least 1 MOhm between each live conductor and earth."))
Q("single_fact", "What is the maximum earth continuity resistance between a panel frame and the earth electrode?",
  "Below 0.5 ohm.",
  (C, "Earth continuity: The resistance between any panel frame and the earth electrode must be below 0.5 ohm."))
Q("single_fact", "What current and voltage are MC4 connectors rated for?",
  "30 A and 1000 V DC.",
  (C, "MC4 connectors are rated for 30 A and 1000 V DC."))
Q("single_fact", "What annual performance ratio does a healthy residential system have?",
  "75-85%.",
  (C, "A healthy residential system has an annual PR of 75-85%."))
Q("single_fact", "How often should monitoring data be logged?",
  "Every 5 minutes.",
  (C, "Data logging interval: 5 minutes"))
Q("single_fact", "When should the monitoring system raise a low-yield alert?",
  "When the daily yield falls below 80% of the expected value for the same irradiance.",
  (C, "raise an alert if the daily yield falls below 80% of the expected value for the same irradiance"))
Q("single_fact", "How often should the insulation resistance test be repeated?",
  "Every 2 years.",
  (C, "Repeat the insulation resistance test every 2 years."))

# =====================================================================================
# C. MULTI-FACT (several facts, one document)
# =====================================================================================
Q("multi_fact", "List every system loss that should be accounted for when estimating solar output.",
  "Inverter efficiency 5-8%, wiring and connector 2-3%, soiling (3-5% arid, 1-2% moderate), temperature derating (0.35% per degree above 25 C), mismatch 1-3%; total 12-20%.",
  (S, "Inverter efficiency loss: 5-8%"), (S, "Wiring and connector losses: 2-3%"),
  (S, "Soiling losses (dust, bird droppings): 3-5% in arid regions, 1-2% in moderate climates"),
  (S, "Temperature derating: approximately 0.35% per degree Celsius above 25 degrees ambient"),
  (S, "Mismatch losses: 1-3%"), (S, "Total expected system loss: 12-20%"))
Q("multi_fact", "What are the six steps of the solar installation procedure?",
  "1 mount racking, 2 attach MC4 connectors, 3 run DC cabling, 4 connect inverter to the distribution board, 5 ground frames and inverter, 6 commission.",
  (S, "Mount the racking system to the roof structure"), (S, "Attach the MC4 connectors to the panel junction boxes."),
  (S, "Run DC cabling (minimum 6 mm²"), (S, "Connect the inverter to the main switchboard at the distribution board."),
  (S, "Ground the frame of each panel and the inverter chassis to the earth electrode."),
  (S, "Commission the system. Power on the inverter"))
Q("multi_fact", "What maintenance tasks are required monthly, quarterly, annually and every 5 years?",
  "Monthly visual inspection; quarterly cleaning with soft brush and deionised water; annually inspect DC/AC connections and check firmware; every 5 years replace MC4 connectors.",
  (S, "Monthly: Visual inspection for physical damage"), (S, "Quarterly: Clean panels with a soft brush and deionised water."),
  (S, "Annually: Inspect all DC and AC connections for corrosion or terminal loosening."),
  (S, "Every 5 years: Replace all DC connectors (MC4) as a precaution"))
Q("multi_fact", "What do the inverter fault codes F-001 to F-005 mean?",
  "F-001 grid voltage out of range; F-002 DC input overvoltage; F-003 inverter over-temperature; F-004 ground fault; F-005 communication failure with the monitoring gateway.",
  (S, "Code F-001: Grid voltage out of range."), (S, "Code F-002: DC input overvoltage."),
  (S, "Code F-003: Inverter over-temperature."), (S, "Code F-004: Ground fault detected."),
  (S, "Code F-005: Communication failure between inverter and monitoring gateway."))
Q("multi_fact", "How does panel orientation affect annual energy yield?",
  "Facing true south (northern hemisphere) or true north (southern hemisphere) gives up to 25% more energy; east-west orientations lose approximately 10-15% in annual yield.",
  (S, "Panels produce up to 25% more energy when facing true south (in the northern hemisphere) or true north (southern hemisphere)."),
  (S, "East-west orientations lose approximately 10-15% in annual yield."))
Q("multi_fact", "How does LiFePO4 cycle life change with depth of discharge?",
  "20% DOD: 8,000+ cycles; 40%: 5,000+; 60%: 3,000+; 80%: 2,000-3,000; 100%: 800-1,200 (not recommended).",
  (B, "20% DOD: 8,000+ cycles"), (B, "40% DOD: 5,000+ cycles"), (B, "60% DOD: 3,000+ cycles"),
  (B, "80% DOD: 2,000-3,000 cycles"), (B, "100% DOD: 800-1,200 cycles (not recommended)"))
Q("multi_fact", "Compare the cycle life of LiFePO4, NMC and lead-acid batteries.",
  "LiFePO4 2,000-3,000 cycles at 80% DOD; NMC 1,000-2,000 cycles at 80% DOD; lead-acid 300-500 cycles at 50% DOD.",
  (B, "Cycle life: 2,000-3,000 cycles at 80% depth of discharge"),
  (B, "Cycle life: 1,000-2,000 cycles at 80% depth of discharge"),
  (B, "Cycle life: 300-500 cycles at 50% depth of discharge"))
Q("multi_fact", "Compare the cost of LiFePO4, NMC and lead-acid batteries.",
  "LiFePO4 cost is moderate, NMC is lower per kWh, and lead-acid has the lowest initial cost but the highest lifetime cost due to replacement.",
  (B, "Cost: Moderate"), (B, "Cost: Lower per kWh"), (B, "Cost: Lowest initial, highest lifetime (due to replacement)"))
Q("multi_fact", "At what cell temperatures must a BMS stop charging, and how are temperatures measured?",
  "Charging must be halted if any cell exceeds 45 degrees C or drops below 0 degrees C; temperature is measured by NTC thermistors on each cell.",
  (B, "NTC thermistors placed on each cell."),
  (B, "Charging must be halted if any cell exceeds 45 degrees C or drops below 0 degrees C."))
Q("multi_fact", "How are cells balanced in a BMS and when is a cell considered defective?",
  "Cells are balanced to within 50 mV of each other; a drift beyond 200 mV indicates an aging or defective cell.",
  (B, "Individual cell balancing to keep all cells within 50 mV of each other."),
  (B, "A drift beyond 200 mV indicates an aging or defective cell."))
Q("multi_fact", "What is the round-trip efficiency and self-discharge of each battery chemistry?",
  "Round-trip efficiency: LiFePO4 92-96%, NMC 90-95%, lead-acid 75-85%. Self-discharge: LiFePO4 2-3% per month, lead-acid 3-5% per month.",
  (B, "Round-trip efficiency: LiFePO4 92-96%, NMC 90-95%, lead-acid 75-85%"),
  (B, "Self-discharge: LiFePO4 2-3% per month, lead-acid 3-5% per month"))
Q("multi_fact", "What are the main differences between string inverters and microinverters?",
  "A string inverter serves series strings, is cheapest per kW and simplest to maintain, but shading on one panel reduces the whole string. A microinverter sits behind each panel so shading affects only that panel, but it costs more per kW and has more units to service.",
  (I, "String inverter: One inverter serves one or more series strings of panels."),
  (I, "Shading on one panel reduces the output of the whole string."),
  (I, "Microinverter: One small inverter is mounted behind each panel."),
  (I, "It has a higher cost per kW and more units to service on the roof."))
Q("multi_fact", "What are the MPPT specifications of a typical 5 kW residential hybrid inverter?",
  "MPPT window 120 V to 500 V, maximum DC input 600 V, 2 MPPT inputs, maximum 15 A per MPPT.",
  (I, "MPPT voltage window: 120 V to 500 V"), (I, "Maximum DC input voltage: 600 V"),
  (I, "Number of MPPT inputs: 2"), (I, "Maximum input current per MPPT: 15 A"))
Q("multi_fact", "Describe the specification of a 5 kW class hybrid inverter.",
  "5 kW continuous at 230 V, 7.5 kW surge for 10 s, 51.2 V nominal battery input, 100 A max battery current, under 20 ms changeover, IP65, -25 to 60 C with derating above 40 C, 15 cm clearance.",
  (I, "Rated AC output: 5 kW continuous, 230 V"), (I, "Surge capacity: 7.5 kW for 10 seconds"),
  (I, "Battery input: 51.2 V nominal (matches a 16S LiFePO4 pack)"),
  (I, "Maximum battery charge and discharge current: 100 A"),
  (I, "Backup changeover time: under 20 ms"), (I, "Enclosure rating: IP65, suitable for outdoor wall mounting"),
  (I, "Ambient operating range: -25 to 60 degrees C, with output derating above 40 degrees C"),
  (I, "Mounting clearance: at least 15 cm on all sides"))
Q("multi_fact", "What additional fault codes do hybrid inverters add beyond F-005?",
  "F-006 battery communication lost; F-007 battery under-voltage (below 44.8 V); F-008 battery over-voltage (above 58.4 V); F-009 AC overload.",
  (I, "Code F-006: Battery communication lost."), (I, "Code F-007: Battery under-voltage. Pack voltage is below 44.8 V."),
  (I, "Code F-008: Battery over-voltage. Pack voltage is above 58.4 V."), (I, "Code F-009: AC overload."))
Q("multi_fact", "What size array and number of panels does a home using 14 kWh per day need at 4.2 peak sun hours?",
  "14 / (4.2 x 0.8) = 4.17 kW, rounded up to 11 panels of 400 Wp, a 4.4 kW array.",
  (Z, "Array size in kW = daily energy target / (peak sun hours x 0.8)."),
  (Z, "Worked example: 14 kWh per day at 4.2 peak sun hours gives 14 / (4.2 x 0.8) = 4.17 kW. Round up to 11 panels of 400 Wp, which is a 4.4 kW array."))
Q("multi_fact", "What nominal battery capacity is needed for a 5.6 kWh evening load at 60% depth of discharge?",
  "5.6 kWh x 1.2 = 6.72 kWh usable; at 60% DOD the nominal capacity is 6.72 / 0.6 = 11.2 kWh.",
  (Z, "Usable capacity needed = evening load x 1.2."),
  (Z, "Worked example: an evening load of 5.6 kWh x 1.2 = 6.72 kWh usable. At 60% depth of discharge the nominal capacity is 6.72 / 0.6 = 11.2 kWh."))
Q("multi_fact", "What is the evening load of a home that uses 14 kWh per day?",
  "At an evening share of 40%, the evening load is 5.6 kWh (evening load is typically 35-45% of daily consumption).",
  (Z, "Worked example: a home uses 14 kWh per day. At an evening share of 40%, the evening load is 5.6 kWh."))
Q("multi_fact", "How many peak sun hours do different regions get?",
  "Northern Europe 2.5-3.0; central Europe and moderate climates 3.5-4.5; Mediterranean and US Southwest 5.0-6.5; equatorial regions 4.5-5.5 hours per day.",
  (Z, "Northern Europe: 2.5-3.0 hours per day"),
  (Z, "Central Europe and other moderate climates: 3.5-4.5 hours per day"),
  (Z, "Mediterranean and US Southwest: 5.0-6.5 hours per day"),
  (Z, "Equatorial regions: 4.5-5.5 hours per day"))
Q("multi_fact", "What are the pre-energisation tests before closing the DC isolator?",
  "Polarity check, open-circuit voltage within 5% of calculated, insulation resistance at 500 V DC of at least 1 MOhm, and earth continuity below 0.5 ohm.",
  (C, "Polarity check: Measure each string with a multimeter before it is connected to the inverter."),
  (C, "Open-circuit voltage: The measured string Voc must be within 5% of the calculated value."),
  (C, "Insulation resistance: Test at 500 V DC."),
  (C, "Earth continuity: The resistance between any panel frame and the earth electrode must be below 0.5 ohm."))
Q("multi_fact", "What alerts should the monitoring system raise?",
  "A low-yield alert if daily yield is below 80% of expected, and a heartbeat alert if the gateway has not reported for 30 minutes.",
  (C, "raise an alert if the daily yield falls below 80% of the expected value for the same irradiance"),
  (C, "raise an alert if the monitoring gateway has not reported for 30 minutes"))
Q("multi_fact", "What are the common causes of low yield in a solar system?",
  "Soiling, new shading such as tree growth, a failed string (check MC4 connectors and string fuse), and inverter derating from high temperature (F-003).",
  (C, "Soiling on the panels"), (C, "New shading, for example tree growth"),
  (C, "A failed string: if one string produces zero power, check its MC4 connectors and its string fuse"),
  (C, "Inverter derating from high temperature (fault code F-003)"))
Q("multi_fact", "What happens at the first power-on of the system?",
  "After all tests pass, close the DC isolator and then the AC breaker; the inverter runs a grid synchronisation check for up to 5 minutes before exporting power.",
  (C, "After all tests pass, close the DC isolator and then the AC breaker."),
  (C, "The inverter performs a grid synchronisation check for up to 5 minutes before it starts exporting power."))
Q("multi_fact", "What critical loads are typical for backup and how much does each draw?",
  "Refrigerator 150 W average, LED lighting 100 W, router and modem 20 W, phone and laptop chargers 60 W.",
  (Z, "Refrigerator: 150 W average"), (Z, "LED lighting: 100 W"),
  (Z, "Router and modem: 20 W"), (Z, "Phone and laptop chargers: 60 W"))

# =====================================================================================
# D. MULTI-DOCUMENT (needs evidence from >= 2 files)
# =====================================================================================
Q("multi_doc", "A 6 kW array feeds a 5 kW hybrid inverter. What is the DC/AC ratio and how much does the array realistically produce per day in a moderate climate?",
  "The DC/AC ratio is 1.2, within the recommended 1.1-1.3. In a moderate climate the 6 kW system realistically produces 20 to 22 kWh per day.",
  (I, "For example, a 6 kW array on a 5 kW inverter has a ratio of 1.2."),
  (S, "Therefore a 6 kW system in a moderate climate should realistically produce between 20 and 22 kWh per day."))
Q("multi_doc", "Will a 10 kWh LiFePO4 battery used at 60% depth of discharge cover an evening load of 5.6 kWh?",
  "It delivers about 6 kWh per day, which just covers 5.6 kWh, but the sizing guide's 20% margin requires 6.72 kWh usable (11.2 kWh nominal at 60% DOD), so 10 kWh is slightly undersized.",
  (B, "A 10 kWh system used daily at 60% DOD will deliver approximately 6 kWh per day"),
  (Z, "Worked example: a home uses 14 kWh per day. At an evening share of 40%, the evening load is 5.6 kWh."),
  (Z, "Worked example: an evening load of 5.6 kWh x 1.2 = 6.72 kWh usable. At 60% depth of discharge the nominal capacity is 6.72 / 0.6 = 11.2 kWh."))
Q("multi_doc", "What IP rating is required for the battery enclosure and for a hybrid inverter mounted outdoors?",
  "Both need IP65: the battery enclosure must be IP65 or higher for outdoor installation and the hybrid inverter enclosure is rated IP65 for outdoor wall mounting.",
  (B, "The battery enclosure must be IP65 rated or higher for outdoor installation."),
  (I, "Enclosure rating: IP65, suitable for outdoor wall mounting"))
Q("multi_doc", "What does fault code F-003 mean and what mounting clearance does the hybrid inverter need?",
  "F-003 is inverter over-temperature; ensure ventilation clearance of at least 15 cm on all sides, which matches the hybrid inverter's mounting clearance of at least 15 cm.",
  (S, "Code F-003: Inverter over-temperature. Ensure ventilation clearance of at least 15 cm on all sides."),
  (I, "Mounting clearance: at least 15 cm on all sides"))
Q("multi_doc", "Does the hybrid inverter's battery input voltage match a 10 kWh LiFePO4 battery pack?",
  "Yes. The inverter's battery input is 51.2 V nominal (16S LiFePO4), and the 10 kWh pack is 16 cells of 3.2 V in series = 51.2 V.",
  (I, "Battery input: 51.2 V nominal (matches a 16S LiFePO4 pack)"),
  (B, "16 x LiFePO4 cells (3.2V nominal each) in series = 51.2V"))
Q("multi_doc", "How do the maximum currents of the hybrid inverter and the battery BMS compare?",
  "The inverter's maximum battery charge and discharge current is 100 A, matching the BMS rating of 100 A continuous (200 A peak).",
  (I, "Maximum battery charge and discharge current: 100 A"),
  (B, "BMS: 16S, 100A continuous, 200A peak"))
Q("multi_doc", "What pack voltage triggers battery under-voltage fault F-007, and what is the nominal voltage of the 16-cell LiFePO4 pack?",
  "F-007 triggers below 44.8 V; the nominal pack voltage of the 16-cell LiFePO4 system is 51.2 V.",
  (I, "Code F-007: Battery under-voltage. Pack voltage is below 44.8 V."),
  (B, "16 x LiFePO4 cells (3.2V nominal each) in series = 51.2V"))
Q("multi_doc", "How many 400 Wp panels does a 6 kW array need and how much does it produce per day in a moderate climate?",
  "15 panels (about 28.5 square metres), producing realistically 20 to 22 kWh per day.",
  (Z, "A 6 kW array therefore needs 15 panels and about 28.5 square metres of roof area."),
  (S, "Therefore a 6 kW system in a moderate climate should realistically produce between 20 and 22 kWh per day."))
Q("multi_doc", "What dead load must the roof support and how much roof area does a 6 kW array occupy?",
  "The roof must support at least 30 kg per square meter dead load; a 6 kW array of 15 panels needs about 28.5 square metres.",
  (S, "Residential roofs must support a dead load of at least 30 kg per square meter for the mounting hardware and panels"),
  (Z, "A 6 kW array therefore needs 15 panels and about 28.5 square metres of roof area."))
Q("multi_doc", "How should polarity be checked before connecting MC4 connectors and what can reverse polarity damage?",
  "Measure each string with a multimeter before it is connected, and correct reversed polarity before mating any MC4 connector to the inverter input; reverse polarity can damage the inverter.",
  (C, "Polarity check: Measure each string with a multimeter before it is connected to the inverter."),
  (C, "Reversed polarity must be corrected before any MC4 connector is mated to the inverter input."),
  (S, "A reverse polarity connection can damage the inverter."))
Q("multi_doc", "What are the MC4 connector ratings and how often should MC4 connectors be replaced?",
  "MC4 connectors are rated 30 A and 1000 V DC and should be replaced every 5 years as a precaution against UV degradation.",
  (C, "MC4 connectors are rated for 30 A and 1000 V DC."),
  (S, "Every 5 years: Replace all DC connectors (MC4) as a precaution, as UV degradation of the polymer housing can lead to arcing."))
Q("multi_doc", "How does cold weather affect string voltage and what maximum DC voltage must it stay below?",
  "Voc rises about 0.3% per degree Celsius below 25 degrees, so string Voc must be calculated at the coldest site temperature and stay below the inverter's maximum DC input voltage (600 V for a typical 5 kW hybrid; typically 600 V or 1000 V).",
  (I, "Open-circuit voltage (Voc) rises as panels get colder, by approximately 0.3% per degree Celsius below 25 degrees."),
  (I, "Maximum DC input voltage: 600 V"),
  (S, "Verify panel string voltage does not exceed inverter maximum DC voltage (typically 600V or 1000V depending on model)."))
Q("multi_doc", "What annual performance ratio is expected, and what total system loss range does the installation guide use?",
  "A healthy system has an annual performance ratio of 75-85%; the installation guide uses a total expected system loss of 12-20%.",
  (C, "A healthy residential system has an annual PR of 75-85%."),
  (S, "Total expected system loss: 12-20%"))
Q("multi_doc", "How does the 5-8% inverter loss in the installation guide relate to inverter efficiency?",
  "A 5-8% inverter loss is consistent with a weighted (European) efficiency of 92-95% for residential inverters.",
  (S, "Inverter efficiency loss: 5-8%"),
  (I, "Weighted (European) efficiency is 92-95% for residential units"))
Q("multi_doc", "Which fault code indicates a ground fault and what earth continuity resistance must be achieved at commissioning?",
  "F-004 indicates a ground fault; commissioning requires earth continuity below 0.5 ohm between any panel frame and the earth electrode.",
  (S, "Code F-004: Ground fault detected. Inspect DC cabling for insulation damage."),
  (C, "Earth continuity: The resistance between any panel frame and the earth electrode must be below 0.5 ohm."))
Q("multi_doc", "How long does a 10 kWh battery used at 60% DOD last, and what output do panels have after 25 years?",
  "The battery lasts around 3,000 days (8 years) before dropping below 80% of rated capacity, so it is replaced before the panels, which retain about 87% of their original output after 25 years.",
  (B, "A 10 kWh system used daily at 60% DOD will deliver approximately 6 kWh per day and last around 3,000 days (8 years) before capacity drops below 80% of rated."),
  (Z, "Panel output degrades by about 0.5% per year. After 25 years expect about 87% of the original output."))
Q("multi_doc", "The installation guide uses 4.2 peak sun hours. What region category in the sizing guide does that fall into?",
  "4.2 peak sun hours falls in the central Europe and other moderate climates category (3.5-4.5 hours per day).",
  (S, "For a 6 kW system at a site with 4.2 peak sun hours per day"),
  (Z, "Central Europe and other moderate climates: 3.5-4.5 hours per day"))
Q("multi_doc", "At what ambient temperature does the hybrid inverter start derating, and what temperature range can LiFePO4 batteries operate in?",
  "The inverter derates its output above 40 degrees C ambient; LiFePO4 batteries operate from 0 to 45 degrees C.",
  (I, "Ambient operating range: -25 to 60 degrees C, with output derating above 40 degrees C"),
  (B, "Operating temperature: 0 to 45 degrees C"))
Q("multi_doc", "Where must the battery DC disconnect be located, and in what order should the DC isolator and AC breaker be closed at first power-on?",
  "The DC disconnect must be clearly labelled and within 2 metres of the battery; at first power-on close the DC isolator first and then the AC breaker.",
  (B, "The DC disconnect switch must be clearly labelled and located within 2 metres of the battery."),
  (C, "After all tests pass, close the DC isolator and then the AC breaker."))
Q("multi_doc", "What array size does a 14 kWh per day home need at 4.2 peak sun hours, and what minimum DC cable cross-section applies for runs under 20 metres?",
  "About 4.17 kW (rounded up to 11 panels of 400 Wp = 4.4 kW); DC cabling for runs under 20 metres must be at least 6 mm2.",
  (Z, "Worked example: 14 kWh per day at 4.2 peak sun hours gives 14 / (4.2 x 0.8) = 4.17 kW. Round up to 11 panels of 400 Wp, which is a 4.4 kW array."),
  (S, "Run DC cabling (minimum 6 mm² cross-section for runs under 20 metres) through a waterproof conduit from the panel array to the inverter location."))
Q("multi_doc", "What round-trip efficiency does LiFePO4 have, and how large a margin does the sizing guide add on the evening load?",
  "LiFePO4 round-trip efficiency is 92-96%; the sizing guide adds a 20% margin (evening load x 1.2) for conversion losses and ageing.",
  (B, "Round-trip efficiency: LiFePO4 92-96%, NMC 90-95%, lead-acid 75-85%"),
  (Z, "Usable capacity needed = evening load x 1.2. The 20% margin covers conversion losses and ageing."))
Q("multi_doc", "Which components need replacing every 5 years and every 7 years?",
  "MC4 DC connectors every 5 years; inverter cooling fans every 7 years.",
  (S, "Every 5 years: Replace all DC connectors (MC4) as a precaution"),
  (I, "Cooling fans: inspect annually and replace every 7 years"))
Q("multi_doc", "How often is monitoring data logged, and what does fault code F-005 mean?",
  "Data is logged every 5 minutes; F-005 is a communication failure between the inverter and monitoring gateway (check RS485 or Wi-Fi link).",
  (C, "Data logging interval: 5 minutes"),
  (S, "Code F-005: Communication failure between inverter and monitoring gateway. Check RS485 or Wi-Fi link."))
Q("multi_doc", "What happens to a 6 kW array's roof area and its structural requirements: how many panels, and who must approve the roof?",
  "A 6 kW array needs 15 panels (about 28.5 square metres), and a licensed structural engineer must sign off on the roof before mounting.",
  (Z, "A 6 kW array therefore needs 15 panels and about 28.5 square metres of roof area."),
  (S, "A licensed structural engineer must sign off before mounting."))
Q("multi_doc", "What fault code appears if the battery BMS loses contact with the inverter, and what cell voltage drift suggests a defective cell?",
  "F-006 (check the CAN bus cable between the BMS and inverter); a cell voltage drift beyond 200 mV indicates an aging or defective cell.",
  (I, "Code F-006: Battery communication lost. Check the CAN bus cable between the battery BMS and the inverter."),
  (B, "A drift beyond 200 mV indicates an aging or defective cell."))

# =====================================================================================
# E. UNANSWERABLE (no gold spans; correct behaviour = abstain)
#    "near-miss" ones sit next to facts the corpus does contain, to catch plausible guessing
# =====================================================================================
for q in [
    "How much does a 6 kW residential solar system cost to install?",
    "What is the warranty period for a Tesla Powerwall?",
    "How do I install a small wind turbine on my roof?",
    "What is the federal tax credit percentage for home solar installations?",
    "What is the cycle life of NMC cells at 100% depth of discharge?",                      # near-miss
    "What is the weight of a 400 Wp solar panel?",                                           # near-miss
    "What torque should the roof mounting screws be tightened to?",                          # near-miss
    "What is the efficiency of monocrystalline versus polycrystalline panels?",
    "How do I connect a diesel generator to a hybrid inverter?",
    "What does fault code F-010 mean?",                                                      # near-miss
    "What is the cell price per kWh for LiFePO4 batteries?",                                 # near-miss
    "What is the warranty period on a hybrid inverter?",                                     # near-miss
    "How are lead-acid batteries recycled?",
    "How do I apply for net metering with my utility company?",
    "What is the operating temperature range of sodium-ion batteries?",
    "Which brand of solar panel is the best?",
    "How many cycles does a solid-state battery last?",
    "What is the MPPT voltage window of a 10 kW three-phase inverter?",                      # near-miss
    "What is the shading loss for a bifacial solar panel?",
    "How long does it take to install a residential solar system?",
    "What is the open-circuit voltage of a 400 Wp panel?",                                   # near-miss
    "What is the feed-in tariff per kWh in Germany?",
    "What is the lifetime of NMC batteries at 50% depth of discharge?",                      # near-miss
    "How many peak sun hours does Tokyo get per day?",                                       # near-miss
    "What is the self-discharge rate of NMC batteries?",                                     # near-miss
    "What is the round-trip efficiency of sodium-ion batteries?",
    "What is the maximum number of panels allowed per MPPT string?",                         # near-miss
    "What minimum insulation resistance is required for AC cables?",                         # near-miss
    "Who won the 2022 FIFA World Cup?",                                                      # off-topic
    "Write me a poem about the sun.",                                                        # off-topic
    "What is the best way to dispose of old solar panels?",
    "Can I run my air conditioner on the 10 kWh battery overnight?",                         # near-miss (no AC load data)
]:
    Q("unanswerable", q, UNANS)

# =====================================================================================
# F. VAGUE / BADLY PHRASED (answerable, but short, sloppy, typo-ridden or underspecified)
# =====================================================================================
Q("vague", "tilt?", "The optimal tilt angle equals the site's latitude (e.g. 40 degrees at 40 degrees north latitude).",
  (S, "The optimal tilt angle equals the latitude of the installation site."))
Q("vague", "how often clean panels", "Quarterly, with a soft brush and deionised water.",
  (S, "Quarterly: Clean panels with a soft brush and deionised water."))
Q("vague", "f003", "Inverter over-temperature; ensure at least 15 cm ventilation clearance on all sides.",
  (S, "Code F-003: Inverter over-temperature. Ensure ventilation clearance of at least 15 cm on all sides."))
Q("vague", "F-002 what do i do", "F-002 is DC input overvoltage. Verify the panel string voltage does not exceed the inverter's maximum DC voltage (typically 600 V or 1000 V).",
  (S, "Code F-002: DC input overvoltage. Verify panel string voltage does not exceed inverter maximum DC voltage (typically 600V or 1000V depending on model)."))
Q("vague", "battery lifetime", "It depends on depth of discharge: LiFePO4 gives 8,000+ cycles at 20% DOD down to 800-1,200 cycles at 100%; a 10 kWh system at 60% DOD lasts about 3,000 days (8 years).",
  (B, "20% DOD: 8,000+ cycles"), (B, "60% DOD: 3,000+ cycles"), (B, "100% DOD: 800-1,200 cycles (not recommended)"))
Q("vague", "is lifepo4 safe", "Yes. LiFePO4 has excellent thermal stability with a non-flammable electrolyte and is best for high-safety, long-duration residential storage.",
  (B, "Thermal stability: Excellent, non-flammable electrolyte"),
  (B, "Best for: Long-duration residential storage, high safety"))
Q("vague", "what wire size for the panels", "DC cabling minimum 6 mm2 for runs under 20 metres; inverter output cable minimum 10 mm2; grounding wire minimum 4 mm2 green/yellow.",
  (S, "Run DC cabling (minimum 6 mm² cross-section for runs under 20 metres)"),
  (S, "The inverter output cable must be minimum 10 mm²"),
  (S, "Grounding wire must be minimum 4 mm² green/yellow insulated cable."))
Q("vague", "how many panels do i need for 6kw", "A 6 kW array of typical 400 Wp panels needs 15 panels and about 28.5 square metres of roof area.",
  (Z, "A 6 kW array therefore needs 15 panels and about 28.5 square metres of roof area."))
Q("vague", "shade bad??", "Yes. Even partial shading on a single cell can reduce the output of an entire string by up to 80%.",
  (S, "Even partial shading on a single cell can reduce the output of an entire string by up to 80%."))
Q("vague", "what does bms do", "The BMS balances cell voltage, monitors temperature (halting charging above 45 C or below 0 C), limits charge and discharge current, and estimates state of charge.",
  (B, "Individual cell balancing to keep all cells within 50 mV of each other."),
  (B, "Charging must be halted if any cell exceeds 45 degrees C or drops below 0 degrees C."),
  (B, "Both charge and discharge current must be limited to the cell datasheet maximum."))
Q("vague", "IP rating battery outdoors", "The battery enclosure must be IP65 rated or higher for outdoor installation.",
  (B, "The battery enclosure must be IP65 rated or higher for outdoor installation."))
Q("vague", "pressure washer ok?", "Not above 40 bar; higher pressure can crack the tempered glass cover.",
  (S, "Never use abrasive cleaners or pressure washers above 40 bar, as these can crack the tempered glass cover."))
Q("vague", "panels get hot what happens to output", "Output derates approximately 0.35% per degree Celsius above 25 degrees ambient.",
  (S, "Temperature derating: approximately 0.35% per degree Celsius above 25 degrees ambient"))
Q("vague", "inverter error grid", "F-001: grid voltage out of range; check the utility connection.",
  (S, "Code F-001: Grid voltage out of range. Check utility connection."))
Q("vague", "how long will 10kwh battery last daily use", "Used daily at 60% DOD it delivers about 6 kWh per day and lasts around 3,000 days (8 years) before capacity drops below 80%.",
  (B, "A 10 kWh system used daily at 60% DOD will deliver approximately 6 kWh per day and last around 3,000 days (8 years) before capacity drops below 80% of rated."))
Q("vague", "mc4 how long do they last", "Replace all MC4 connectors every 5 years because UV degradation of the housing can lead to arcing.",
  (S, "Every 5 years: Replace all DC connectors (MC4) as a precaution, as UV degradation of the polymer housing can lead to arcing."))
Q("vague", "cheapest battery type", "Lead-acid has the lowest initial cost but the highest lifetime cost due to replacement; NMC is lower cost per kWh than LiFePO4, which is moderate.",
  (B, "Cost: Moderate"), (B, "Cost: Lower per kWh"), (B, "Cost: Lowest initial, highest lifetime (due to replacement)"))
Q("vague", "is my roof strong enough", "Residential roofs need a dead load capacity of at least 30 kg per square meter, plus a live load of 1.5 kN per square meter in snow regions, and a licensed structural engineer must sign off.",
  (S, "Residential roofs must support a dead load of at least 30 kg per square meter for the mounting hardware and panels"),
  (S, "A licensed structural engineer must sign off before mounting."))
Q("vague", "fire extinguisher battery", "A Class D fire extinguisher must be available within 3 metres.",
  (B, "A Class D fire extinguisher must be available within 3 metres."))
Q("vague", "how much power per day from my 6kw", "25.2 kWh per day before losses at 4.2 peak sun hours; realistically 20 to 22 kWh per day in a moderate climate.",
  (S, "the expected daily generation is 6,000 W x 4.2 h = 25.2 kWh per day, before system losses."),
  (S, "Therefore a 6 kW system in a moderate climate should realistically produce between 20 and 22 kWh per day."))
Q("vague", "inverter too small for array?", "A DC/AC ratio between 1.1 and 1.3 is recommended; for example a 6 kW array on a 5 kW inverter is 1.2, and above 1.3 clipping losses become significant.",
  (I, "A ratio between 1.1 and 1.3 is recommended."),
  (I, "Above 1.3, clipping losses become significant"))
Q("vague", "whats mppt", "The maximum power point tracker adjusts the operating voltage of each string to extract the most power.",
  (I, "The maximum power point tracker (MPPT) adjusts the operating voltage of each string to extract the most power."))
Q("vague", "battery comm error", "F-006: battery communication lost; check the CAN bus cable between the battery BMS and the inverter.",
  (I, "Code F-006: Battery communication lost. Check the CAN bus cable between the battery BMS and the inverter."))
Q("vague", "how big battery for evenings", "Usable capacity needed = evening load x 1.2, and nominal capacity = usable capacity / planned depth of discharge (e.g. 5.6 kWh evening load needs 6.72 kWh usable, 11.2 kWh nominal at 60% DOD).",
  (Z, "Usable capacity needed = evening load x 1.2."),
  (Z, "Nominal capacity = usable capacity / planned depth of discharge."))
Q("vague", "insulation test value", "Test at 500 V DC; at least 1 MOhm between each live conductor and earth.",
  (C, "Insulation resistance: Test at 500 V DC. The result must be at least 1 MOhm between each live conductor and earth."))
Q("vague", "panel output after many years", "Output degrades about 0.5% per year; after 25 years expect about 87% of the original output.",
  (Z, "Panel output degrades by about 0.5% per year. After 25 years expect about 87% of the original output."))
Q("vague", "south facing better?", "Yes. In the northern hemisphere panels facing true south produce up to 25% more energy.",
  (S, "Panels produce up to 25% more energy when facing true south (in the northern hemisphere) or true north (southern hemisphere)."))
Q("vague", "cold weather panels voltage", "Voc rises approximately 0.3% per degree Celsius below 25 degrees, so string Voc must be calculated at the coldest site temperature and stay below the inverter's maximum DC input voltage.",
  (I, "Open-circuit voltage (Voc) rises as panels get colder, by approximately 0.3% per degree Celsius below 25 degrees."))
Q("vague", "can i unplug mc4 while running", "No. Never unplug an MC4 connector while the string is under load; open the DC isolator first.",
  (C, "Never unplug an MC4 connector while the string is under load; open the DC isolator first."))
Q("vague", "battery charging temp limits", "Charging must be halted if any cell exceeds 45 degrees C or drops below 0 degrees C.",
  (B, "Charging must be halted if any cell exceeds 45 degrees C or drops below 0 degrees C."))
Q("vague", "lead acid vs lifepo4", "LiFePO4 offers 90-160 Wh/kg and 2,000-3,000 cycles at 80% DOD at moderate cost; lead-acid offers 30-40 Wh/kg and 300-500 cycles at 50% DOD, the lowest initial but highest lifetime cost.",
  (B, "Energy density: 90-160 Wh/kg"), (B, "Cycle life: 2,000-3,000 cycles at 80% depth of discharge"),
  (B, "Energy density: 30-40 Wh/kg"), (B, "Cycle life: 300-500 cycles at 50% depth of discharge"))


def find_span(doc_text, span):
    pat = r"\s+".join(re.escape(t) for t in span.split())
    return re.search(pat, doc_text)


def split_of(qid):
    h = int(hashlib.md5(f"{SPLIT_SALT}:{qid}".encode()).hexdigest(), 16) % 1000
    return "heldout" if h < HELDOUT_FRACTION * 1000 else "tuning"


def main():
    docs = {p.name: p.read_text(encoding="utf-8") for p in DATA_DIR.glob("*.txt")}
    problems = []
    seen_q = set()
    for n, r in enumerate(rows, 1):
        r["id"] = f"q{n:03d}"
        norm = re.sub(r"\W+", " ", r["question"].lower()).strip()
        if norm in seen_q:
            problems.append(f"{r['id']}: duplicate question")
        seen_q.add(norm)
        if r["category"] == "unanswerable" and r["gold"]:
            problems.append(f"{r['id']}: unanswerable question has gold spans")
        if r["category"] != "unanswerable" and not r["gold"]:
            problems.append(f"{r['id']}: answerable question has no gold span")
        if r["category"] == "multi_doc" and len({g["source"] for g in r["gold"]}) < 2:
            problems.append(f"{r['id']}: multi_doc question uses a single source")
        for g in r["gold"]:
            if g["source"] not in docs:
                problems.append(f"{r['id']}: unknown source {g['source']}")
            elif not find_span(docs[g["source"]], g["text"]):
                problems.append(f"{r['id']}: span not found in {g['source']}: {g['text'][:70]!r}")

    if problems:
        print("\n".join(problems))
        sys.exit(f"{len(problems)} problem(s) - fix before writing the test set")

    # Stratified check: hash split can be uneven in small categories, report it.
    for r in rows:
        r["split"] = split_of(r["id"])

    with OUT.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"wrote {len(rows)} questions to {OUT}")
    tab = Counter((r["category"], r["split"]) for r in rows)
    for cat in ["single_fact", "multi_fact", "multi_doc", "unanswerable", "vague"]:
        print(f"  {cat:13s} tuning={tab[(cat, 'tuning')]:3d}  heldout={tab[(cat, 'heldout')]:3d}")
    print(f"  total tuning={sum(v for (c, s), v in tab.items() if s == 'tuning')}"
          f"  heldout={sum(v for (c, s), v in tab.items() if s == 'heldout')}")


if __name__ == "__main__":
    main()
