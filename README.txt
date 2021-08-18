
In this folder you find the pairwise date responses of all date interactions.

For  clarity,  when  addressing  the  speed  dates  we  treat date as the information from a single
person during a 3 minute date,  and date  interaction as  the  interaction  between  2 participants during a 3 minute speed date.

Each ID is composed as follows:

Digit:
1 - Sex (males = 1, females = 2)
2 - Event Day
3 & 4 - Participant Number per day (no global)

Thus, participant with ID 1121 will be participant 21 of day 1, which is a male. Same format it is used for the PP_ID, or partner ID. This is the person with whom participant ID is having this date interaction. This is summarize in Date_ID, which fuses both ID and PP_ID.

All columns of the file are:
-Date_ID: ID of the date interaction, which is a concatenation of ID and PP_ID
-ID: participant ID following the format described above.
Part_Num: Participant Number per day (no global)
-Event: day of event (1, 2 or 3).
-Group size: total number of people for that event.
-Date_order: order in which each date took place (1=first date of the night for these two people, and so on).
-M_1: Response to question "Have you met this person before today?" Yes=1, No=0.
-M_2: Response to question "Do you want to see this person again?"  Yes=1, No=0.
-M_3: Response to question "How much would you like to see this person again?" Scale 0(low) to 7.
-M_4: Response to question "How would you rate this person as a potential friend?" Scale 0(low) to 7.
-M_5: Response to question "How would you rate this person as a a short term sexual partner?" Scale 0(low) to 7.
-M_6: Response to question "How would you rate this person as a long term romantic partner?" Scale 0(low) to 7.
-Sex: sex of this participant (male = 1, female= 2).
-PP_ID: ID of the partner (e.g. person with whom this participant is having the date), following the same format described above.
-PP_M_1: response M_1 of partner.
-PP_M_2: response M_2 of partner.
-PP_M_3: response M_3 of partner.
-PP_M_4: response M_4 of partner.
-PP_M_5: response M_5 of partner.
-PP_M_6: response M_6 of partner.
-PP_sex: sex of partner.

NOTE: if value 999 appears, it means that this response was unreadable from the booklet.

