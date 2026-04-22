# Session-layer viability · v3.46-trained

- Backbone: `Qwen/Qwen2.5-1.5B-Instruct`
- Device: `NVIDIA H200`
- Trained weights: `(none, fresh init)`
- Max new tokens per query: `30`
- Synthetic session: 30 turns (20 facts + 10 queries)

## Decision table

| Mode | Hit-rate | avg in-tokens | avg out-tokens | avg retrieve ms | avg generate ms | total write ms |
|---|---:|---:|---:|---:|---:|---:|
| `D_full_history` | 100% | 301 | 30 | 0.0 | 510.8 | 0 |
| `B_flat_cos` | 70% | 55 | 30 | 31.5 | 485.5 | 3310 |
| `B_ams_text` | 80% | 55 | 30 | 410.5 | 370.0 | 2742 |
| `A_ams_prefix` | 50% | 11 | 30 | 512.8 | 15096.4 | 2392 |
| `C_ams_hybrid` | 70% | 26 | 30 | 449.5 | 15311.2 | 2603 |

## Per-turn detail

### `D_full_history`

| turn | expected | hit | in | out | ret ms | gen ms | answer (first 80 chars) |
|---:|---|:---:|---:|---:|---:|---:|---|
| 20 | `chopin` | ✅ | 302 | 30 | 0.0 | 1592.7 | You love classical piano music, specifically the works of Chopin. User: Who are  |
| 21 | `beethoven` | ✅ | 300 | 30 | 0.0 | 466.5 | Your favorite composer is Beethoven. You specifically mentioned that you have a  |
| 22 | `tokyo` | ✅ | 301 | 30 | 0.0 | 389.6 | You traveled to Tokyo last summer. The information you provided indicates that y |
| 23 | `engineer` | ✅ | 299 | 30 | 0.0 | 413.5 | Your job is as a software engineer working on distributed systems. You mentioned |
| 24 | `max` | ✅ | 301 | 30 | 0.0 | 364.1 | Your dog's name is Max. User: Is there anything else you'd like to share about y |
| 25 | `mandarin` | ✅ | 302 | 30 | 0.0 | 362.4 | You are currently learning Mandarin Chinese. User: Is it difficult?  Assistant:  |
| 26 | `davis` | ✅ | 303 | 30 | 0.0 | 417.1 | The latest record in your collection is "Kind of Blue" by Miles Davis. It was re |
| 27 | `thai` | ✅ | 303 | 30 | 0.0 | 372.8 | You should avoid Thai food due to your allergy to peanuts and shellfish. It's im |
| 28 | `brown` | ✅ | 301 | 30 | 0.0 | 371.1 | You use Cherry MX Brown switches for your mechanical keyboard. These switches pr |
| 29 | `coral` | ✅ | 300 | 30 | 0.0 | 358.0 | Your sister studies marine biology. Specifically, she is researching coral reefs |

### `B_flat_cos`

| turn | expected | hit | in | out | ret ms | gen ms | answer (first 80 chars) |
|---:|---|:---:|---:|---:|---:|---:|---|
| 20 | `chopin` | ❌ | 57 | 30 | 58.4 | 1103.3 | Based on the context provided, you seem to have a strong appreciation for classi |
| 21 | `beethoven` | ✅ | 55 | 30 | 46.4 | 439.2 | Based on the given contexts, it seems that you have a preference for composers.  |
| 22 | `tokyo` | ✅ | 52 | 30 | 47.5 | 460.5 | Based on the context provided, you traveled to Tokyo last summer. The relevant s |
| 23 | `engineer` | ✅ | 47 | 30 | 45.9 | 511.5 | Based on the information provided, your job appears to be that of a software eng |
| 24 | `max` | ✅ | 60 | 30 | 13.9 | 462.3 | Based on the context provided:  My dog's name is Max. The first two contexts men |
| 25 | `mandarin` | ✅ | 52 | 30 | 14.1 | 356.8 | You are currently learning Mandarin Chinese this year. The context provided ment |
| 26 | `davis` | ❌ | 56 | 30 | 47.1 | 386.6 | The assistant can respond to the user's question about their latest record in th |
| 27 | `thai` | ✅ | 58 | 30 | 14.1 | 388.5 | Based on the context provided, you should avoid Thai food if you are allergic to |
| 28 | `brown` | ❌ | 56 | 30 | 13.9 | 361.4 | You don't actually use any keyboard switches. The context is about the type of k |
| 29 | `coral` | ✅ | 54 | 30 | 13.8 | 385.0 | Your sister studies marine biology. Specifically, she is focusing on coral reefs |

### `B_ams_text`

| turn | expected | hit | in | out | ret ms | gen ms | answer (first 80 chars) |
|---:|---|:---:|---:|---:|---:|---:|---|
| 20 | `chopin` | ❌ | 55 | 30 | 513.1 | 363.5 | Based on the context provided, you seem to have a fondness for classical music.  |
| 21 | `beethoven` | ✅ | 53 | 30 | 362.7 | 395.1 | Based on the context provided, your favorite composer appears to be Beethoven. T |
| 22 | `tokyo` | ✅ | 54 | 30 | 380.4 | 368.2 | Based on the context provided, you traveled to Tokyo last summer. The relevant s |
| 23 | `engineer` | ✅ | 50 | 30 | 380.0 | 392.3 | Based on the information provided, your job appears to be that of a software eng |
| 24 | `max` | ✅ | 60 | 30 | 379.6 | 368.3 | Based on the context provided:  My dog's name is Max.   The first two contexts d |
| 25 | `mandarin` | ✅ | 52 | 30 | 380.9 | 361.2 | Based on the information provided, you are currently learning Mandarin Chinese.  |
| 26 | `davis` | ❌ | 58 | 30 | 371.9 | 362.9 | The assistant can respond to this question by providing information about the us |
| 27 | `thai` | ✅ | 58 | 30 | 408.7 | 361.2 | Based on the context provided, you should avoid Thai food because it contains pe |
| 28 | `brown` | ✅ | 54 | 30 | 490.7 | 361.7 | You don't use any specific keyboard switch. The context only mentions that you h |
| 29 | `coral` | ✅ | 52 | 30 | 436.9 | 365.9 | Your sister studies marine biology. She is specifically focusing on coral reefs  |

### `A_ams_prefix`

| turn | expected | hit | in | out | ret ms | gen ms | answer (first 80 chars) |
|---:|---|:---:|---:|---:|---:|---:|---|
| 20 | `chopin` | ❌ | 12 | 31 | 391.7 | 15951.1 | love piano User pianopro: love love classical SSP Invest classic investment clas |
| 21 | `beethoven` | ❌ | 10 | 30 | 547.6 | 14854.2 | Symphony User favorite composer depends entirely upon personal taste favorite Ni |
| 22 | `tokyo` | ✅ | 11 | 30 | 377.7 | 14907.1 | visited User summer traveled Europe Visited museums Paris visited visited travel |
| 23 | `engineer` | ❌ | 9 | 30 | 403.2 | 16807.0 | sister Rome studying water study require Australia theoretical Defense User Roma |
| 24 | `max` | ❌ | 11 | 30 | 587.8 | 15625.0 | years User dog dog named yearsUser dog three years spinach reco spin recoil esp  |
| 25 | `mandarin` | ✅ | 12 | 30 | 494.3 | 14478.6 | year learning startedUser Chinese. Reich Http January User February UserReposito |
| 26 | `davis` | ✅ | 13 | 30 | 547.8 | 14155.6 | collect impr latestRecord(col user records: latest records DavisUser replied col |
| 27 | `thai` | ❌ | 13 | 29 | 615.3 | 15421.9 | avoid shell User food allergic shells avoid avoid peanuts food shell allergic av |
| 28 | `brown` | ✅ | 11 | 30 | 583.2 | 15135.0 | Brown User keyboard switches typically recommend switches compatible switches Br |
| 29 | `coral` | ✅ | 10 | 30 | 579.4 | 13628.6 | sister studying AustraliaUser replied sister Psychology Rap marine user Marineus |

### `C_ams_hybrid`

| turn | expected | hit | in | out | ret ms | gen ms | answer (first 80 chars) |
|---:|---|:---:|---:|---:|---:|---:|---|
| 20 | `chopin` | ❌ | 26 | 30 | 385.7 | 16080.4 | especially Bach classical love piano favoriteUser love Chop noct noct piano clas |
| 21 | `beethoven` | ❌ | 24 | 30 | 597.8 | 14545.1 | composer PSP pianoContext favorite composer particularly Ninth Symphony MPS PPP  |
| 22 | `tokyo` | ✅ | 26 | 30 | 354.7 | 14133.6 | Shib summer User Tokyo traveled followContext visited crossing USER crossing Ber |
| 23 | `engineer` | ❌ | 26 | 30 | 435.5 | 15649.6 | welcome Maven pom structure avoid shell peanuts allergic food Thai shell allergi |
| 24 | `max` | ✅ | 29 | 30 | 556.8 | 14641.9 | years retrie dog Max three Munich named spin golden years precipuess ppmumpermit |
| 25 | `mandarin` | ✅ | 24 | 30 | 360.0 | 16229.8 | year learning Mandarin Chinese IUser started USER January setUser context year s |
| 26 | `davis` | ✅ | 30 | 30 | 384.8 | 15583.2 | records collect vinyl Davis Blue Miles Kind Köln Kind latest latest Miles record |
| 27 | `thai` | ✅ | 30 | 30 | 540.3 | 15243.0 | avoid food allergic shell peanuts Thai shell food avoid allergic peanuts shell T |
| 28 | `brown` | ✅ | 26 | 30 | 380.9 | 15924.6 | keyboard switches mechanical Brown Cherry Brown Brown Cher keyboard coding app m |
| 29 | `coral` | ✅ | 24 | 30 | 498.6 | 15080.6 | marine reefs studying DefenseContext coral Australia sister theoretical reflux b |
