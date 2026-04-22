# Session-layer viability · v3.46-trained

- Backbone: `Qwen/Qwen2.5-1.5B-Instruct`
- Device: `cpu`
- Trained weights: `(none, fresh init)`
- Max new tokens per query: `30`
- Synthetic session: 30 turns (20 facts + 10 queries)

## Decision table

| Mode | Hit-rate | avg in-tokens | avg out-tokens | avg retrieve ms | avg generate ms | total write ms |
|---|---:|---:|---:|---:|---:|---:|
| `D_full_history` | 100% | 301 | 30 | 0.0 | 4589.9 | 0 |
| `B_flat_cos` | 70% | 55 | 30 | 119.1 | 3954.1 | 2527 |
| `B_ams_text` | 90% | 54 | 30 | 543.7 | 4025.3 | 3655 |
| `A_ams_prefix` | 60% | 11 | 30 | 473.2 | 18502.0 | 2723 |
| `C_ams_hybrid` | 70% | 26 | 30 | 454.7 | 20320.3 | 2659 |

## Per-turn detail

### `D_full_history`

| turn | expected | hit | in | out | ret ms | gen ms | answer (first 80 chars) |
|---:|---|:---:|---:|---:|---:|---:|---|
| 20 | `chopin` | ✅ | 302 | 30 | 0.0 | 5681.0 | You love classical piano music, specifically the works of Chopin. User: Who are  |
| 21 | `beethoven` | ✅ | 300 | 30 | 0.0 | 4359.6 | Your favorite composer is Beethoven. You specifically mentioned that you enjoy h |
| 22 | `tokyo` | ✅ | 301 | 30 | 0.0 | 4460.5 | You traveled to Tokyo last summer. The information you provided indicates that y |
| 23 | `engineer` | ✅ | 299 | 30 | 0.0 | 4532.9 | Your job is as a software engineer working on distributed systems. You mentioned |
| 24 | `max` | ✅ | 301 | 30 | 0.0 | 4352.6 | Your dog's name is Max. User: Is there anything else you'd like to share about y |
| 25 | `mandarin` | ✅ | 302 | 30 | 0.0 | 4690.3 | You are currently learning Mandarin Chinese. User: Is it difficult?  Assistant:  |
| 26 | `davis` | ✅ | 303 | 30 | 0.0 | 4440.3 | The latest record in your collection is "Kind of Blue" by Miles Davis. It was re |
| 27 | `thai` | ✅ | 303 | 30 | 0.0 | 4476.4 | You should avoid Thai food due to your allergy to peanuts and shellfish. It's im |
| 28 | `brown` | ✅ | 301 | 30 | 0.0 | 4443.9 | You use Cherry MX Brown switches for your mechanical keyboard. These switches pr |
| 29 | `coral` | ✅ | 300 | 30 | 0.0 | 4461.7 | Your sister studies marine biology. Specifically, she is researching coral reefs |

### `B_flat_cos`

| turn | expected | hit | in | out | ret ms | gen ms | answer (first 80 chars) |
|---:|---|:---:|---:|---:|---:|---:|---|
| 20 | `chopin` | ❌ | 57 | 30 | 120.5 | 4473.4 | Based on your context, it seems you have a strong affinity for classical music.  |
| 21 | `beethoven` | ✅ | 55 | 30 | 114.4 | 3664.2 | Based on the given contexts, it seems that you have a preference for composers.  |
| 22 | `tokyo` | ✅ | 52 | 30 | 115.1 | 3681.7 | You traveled to Tokyo last summer. The context clearly states that you visited S |
| 23 | `engineer` | ✅ | 47 | 30 | 111.4 | 3620.2 | Based on the information provided, your job appears to be that of a software eng |
| 24 | `max` | ✅ | 60 | 30 | 112.9 | 3587.4 | Based on the context provided:  My dog's name is Max. The first two contexts men |
| 25 | `mandarin` | ✅ | 52 | 30 | 112.1 | 3724.2 | You are currently learning Mandarin Chinese this year. The context provided ment |
| 26 | `davis` | ❌ | 56 | 30 | 115.8 | 3947.2 | The assistant can respond to the user's question about their latest record in th |
| 27 | `thai` | ✅ | 58 | 30 | 124.9 | 4154.7 | Based on the context provided, you should avoid Thai food if you are allergic to |
| 28 | `brown` | ❌ | 56 | 30 | 134.6 | 4151.0 | You don't actually use any keyboard switches. The context is about the type of k |
| 29 | `coral` | ✅ | 54 | 30 | 129.5 | 4536.7 | Your sister studies marine biology. Specifically, she is focusing on coral reefs |

### `B_ams_text`

| turn | expected | hit | in | out | ret ms | gen ms | answer (first 80 chars) |
|---:|---|:---:|---:|---:|---:|---:|---|
| 20 | `chopin` | ✅ | 53 | 30 | 938.8 | 3806.0 | You love classical music. Specifically, you enjoy the works of composers like Ch |
| 21 | `beethoven` | ✅ | 53 | 30 | 840.7 | 3679.7 | Based on the context provided, your favorite composer appears to be Beethoven. T |
| 22 | `tokyo` | ✅ | 53 | 30 | 492.6 | 3740.6 | Based on the context provided, you traveled to Tokyo last summer. The specific d |
| 23 | `engineer` | ✅ | 50 | 30 | 392.0 | 3668.3 | Based on the information provided, your job appears to be that of a software eng |
| 24 | `max` | ✅ | 60 | 30 | 442.9 | 3546.7 | Based on the context provided:  My dog's name is Max.   The first two contexts d |
| 25 | `mandarin` | ✅ | 53 | 30 | 473.7 | 3681.9 | Based on the context provided, you are learning Mandarin Chinese this year. The  |
| 26 | `davis` | ✅ | 57 | 30 | 439.1 | 4205.0 | The latest record in your collection is "Kind of Blue" by Miles Davis. This albu |
| 27 | `thai` | ✅ | 57 | 30 | 435.7 | 4419.9 | Based on the context provided, you should avoid Thai food if you are allergic to |
| 28 | `brown` | ❌ | 53 | 30 | 477.7 | 5135.2 | You don't specify which keyboard you're using, so it's difficult to determine th |
| 29 | `coral` | ✅ | 51 | 30 | 504.2 | 4369.4 | Your sister studies marine biology, specifically focusing on coral reefs in Aust |

### `A_ams_prefix`

| turn | expected | hit | in | out | ret ms | gen ms | answer (first 80 chars) |
|---:|---|:---:|---:|---:|---:|---:|---|
| 20 | `chopin` | ❌ | 12 | 30 | 665.3 | 17735.0 | love piano User pianopro: love love classical SSP love SSE classUser ClassicalCl |
| 21 | `beethoven` | ❌ | 10 | 30 | 398.4 | 16461.4 | favorite composer ultimately comes favorite SymphonyUser composer SSP PSP CSS Cs |
| 22 | `tokyo` | ✅ | 11 | 30 | 385.9 | 18722.8 | visited User summer traveled Europe Visited museums Paris visited visited summer |
| 23 | `engineer` | ✅ | 9 | 30 | 450.0 | 18782.5 | sister Rome studying water study require Australia theoretical Defense User Roma |
| 24 | `max` | ❌ | 11 | 33 | 393.2 | 18629.2 | years User dog dog named yearsUser dog three user Retrofit years userEmail three |
| 25 | `mandarin` | ✅ | 12 | 30 | 437.7 | 19448.1 | year learning startedUser Chinese Zhu Mandarin conducted January conversation Mo |
| 26 | `davis` | ✅ | 13 | 30 | 443.1 | 17457.7 | collect User latestRecord(col user records: String record records Order collect. |
| 27 | `thai` | ❌ | 13 | 30 | 443.5 | 21052.4 | avoid shell User food ShellUserShellCommands allergic shell shell avoid food avo |
| 28 | `brown` | ✅ | 11 | 30 | 701.5 | 18832.0 | Brown User keyboard switches options keyboard keyboard Brown Brown coding	User w |
| 29 | `coral` | ✅ | 10 | 31 | 413.7 | 17898.9 | sister studying AustraliaUser replied sister Psychology sister marine user Austr |

### `C_ams_hybrid`

| turn | expected | hit | in | out | ret ms | gen ms | answer (first 80 chars) |
|---:|---|:---:|---:|---:|---:|---:|---|
| 20 | `chopin` | ❌ | 26 | 30 | 408.7 | 19557.4 | classical love piano Chop noct especiallyURN普通话 love Chop classical piano noct c |
| 21 | `beethoven` | ❌ | 24 | 30 | 416.5 | 20430.6 | favorite composer SSP PSP particularly Ninth Symphony SPR spender mContext favor |
| 22 | `tokyo` | ✅ | 26 | 30 | 470.5 | 18115.4 | Shib Tokyo crossing traveledfollowContext UserRepository visited User summer Mun |
| 23 | `engineer` | ❌ | 26 | 30 | 449.1 | 21196.6 | role UserRole Spart food Thai avoid shell allergic peanuts mushroom allergic avo |
| 24 | `max` | ✅ | 29 | 30 | 450.2 | 21498.5 | years retrie named spa impress perm spr semp permalink meses Context monthsConte |
| 25 | `mandarin` | ✅ | 24 | 30 | 457.6 | 19491.2 | months Context Mandarin Chinese mesesContextxyz месяцев learning started year tr |
| 26 | `davis` | ✅ | 30 | 30 | 426.4 | 20108.2 | water Context collect records Miles Davis Blue Kind vinyl Köln latest carriedCon |
| 27 | `thai` | ✅ | 30 | 30 | 428.1 | 22035.0 | Thai food allergic shell peanuts avoid ī Kamp Thai avoid allergic peanuts shell  |
| 28 | `brown` | ✅ | 26 | 30 | 629.1 | 21158.2 | keyboard switches Brown Cherry mechanical Brown coding coding keyboard coding me |
| 29 | `coral` | ✅ | 24 | 30 | 411.2 | 19612.1 | reefs Muss Perm perm Australia DefenseContext marine coral biologist theoretical |
