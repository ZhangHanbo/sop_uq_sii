# SOP-UQ 公开业务判定规则

各域 SOP 和工具参数见同目录子文件夹。下列补充优先于原 SOP 中有歧义的文字和示例。本题分数是基准任务定义，不用作真实业务安全决策。上游 SOP 为语言模型合成文本，并非实际机构流程；本题状态另行生成，不声称真实法规合规、适航安全或工业发生率。

共同约定：工具观测可能不准确或相互依赖；普通工具和 oracle 的访问阶段、可核验字段与预算由服务端执行。训练与测试都不提供完整真实状态；oracle 仅按规则返回获准的单项核验结果。返回结果只能包含 `TARGET_FIELDS` 对应的主任务字段。

工具参数对象只接受对应 schema 的 `properties` 所列字段，`required` 标出必填字段；额外参数在执行前拒绝。实体编号须引用本题公开输入，会话和工单须使用本题此前工具返回的值。

## dangerous_goods

The public source text supplies business context; it is not a clean score table.
Read scoring tools to obtain observations. Scores are integers 1..5; 0/null means missing.
An unregistered product (product_not_found) or invalid reference gives Unable to Decide.
With more than two missing scores use Unable to Decide. Otherwise impute each missing
score with the maximum observed valid score, and sum all four. Sum 4..7 => Hazard Class A,
8..12 => Hazard Class B, 13..16 => Hazard Class C, 17..20 => Hazard Class D.
The sole output target is hazard_class with those values or Unable to Decide.

## aircraft_inspection

This is a synthetic teaching simulator. The upstream SOP was LLM-generated; no real-world regulatory compliance or aviation safety is certified.
aircraft_ready is the administrative maintenance clearance returned by
VerifyAircraftClearance, not a conjunction of the other checks. Mechanical success requires
weight within +/-2% and matching serial numbers; these are necessary but not sufficient.
Cross-check succeeds exactly when absolute component-weight deviation is <= 2% of the expected weight, installation-to-inspection age is in [0, 24 hours], and expected/installed serial numbers match. All boundaries are inclusive.
Read all seven services including the three report services. Report reads retrieve existing
outcomes and do not create reports; report applicability is distinct from a service return.
updateMaintenanceRecord records the SOP maintenance update. Return exactly:
aircraft_ready (JSON boolean), mechanical_inspection_result and electrical_inspection_result
(success/fail/retest), component_incident_response, component_mismatch_response,
cross_check_response and cross_check_reporting_response (success/failed/null).
Do not add the unrelated VerifyShipment field from the illustrative SOP example.

## customer_service

Invalid account, failed authentication or ineligibility => FAILED.
An outage within 10 miles => PENDING_ACTION. For non-payment suspension first call
transferAccountsPayable, check the existing payment and suspension release records;
do not simulate a payment. A diagnostic observation requires the prior eligible-account
and outage checks. At least one abnormal pre KPI is needed for diagnosis.
Let D(L,J,B;P)=max(L/100-1,0)+max(J/30-1,0)+max(1-B/P,0), where P is subscribed Mbps.
After troubleshooting use RESOLVED exactly when Dpre>0, Dpost<=0.5*Dpre,
Lpost<=Lpre, Jpost<=Jpre and Bpost>=Bpre; otherwise ESCALATED.
Values lie on a 0.1 grid and equality qualifies. There is no extra initial-good early stop.
executeTroubleshooting must supply exactly the distinct root_causes returned by the prior diagnostic call; wrong, omitted or duplicate causes are rejected before repair. A canonical repair changes pre to post once; repeats do not repair again. Root cause does
not uniquely determine repair success. Escalation returns a request receipt, not proof
of necessity. Choose the escalation team by issue nature and record status with updateTicket.
The sole main target is final_resolution_status: FAILED/PENDING_ACTION/RESOLVED/ESCALATED.
