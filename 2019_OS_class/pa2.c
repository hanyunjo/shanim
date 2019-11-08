/**********************************************************************
 * Copyright (c) 2019
 *  Sang-Hoon Kim <sanghoonkim@ajou.ac.kr>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 2 as
 * published by the Free Software Foundation.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTIABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 **********************************************************************/

/* THIS FILE IS ALL YOURS; DO WHATEVER YOU WANT TO DO IN THIS FILE */

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include "types.h"
#include "list_head.h"

/**
 * The process which is currently running
 */
#include "process.h"
extern struct process *current;

/**
 * List head to hold the processes ready to run
 */
extern struct list_head readyqueue;


/**
 * Resources in the system.
 */
#include "resource.h"
extern struct resource resources[NR_RESOURCES];


/**
 * Monotonically increasing ticks
 */
extern unsigned int ticks;


/**
 * Quiet mode. True if the program was started with -q option
 */
extern bool quiet;


/***********************************************************************
 * Default FCFS resource acquision function
 *
 * DESCRIPTION
 *   This is the default resource acquision function which is called back
 *   whenever the current process is to acquire resource @resource_id.
 *   The current implementation serves the resource in the requesting order
 *   without considering the priority. See the comments in sched.h
 ***********************************************************************/
bool fcfs_acquire(int resource_id)
{
	struct resource *r = resources + resource_id;

	if (!r->owner) {
		/* This resource is not owned by any one. Take it! */
		r->owner = current;
		return true;
	}

	/* OK, this resource is taken by @r->owner. */

	/* Update the current process state */
	current->status = PROCESS_WAIT;

	/* And append current to waitqueue */
	list_add_tail(&current->list, &r->waitqueue);

	/**
	 * And return false to indicate the resource is not available.
	 * The scheduler framework will soon call schedule() function to
	 * schedule out current and to pick the next process to run.
	 */
	return false;
}

/***********************************************************************
 * Default FCFS resource release function
 *
 * DESCRIPTION
 *   This is the default resource release function which is called back
 *   whenever the current process is to release resource @resource_id.
 *   The current implementation serves the resource in the requesting order
 *   without considering the priority. See the comments in sched.h
 ***********************************************************************/
void fcfs_release(int resource_id)
{
	struct resource *r = resources + resource_id;

	/* Ensure that the owner process is releasing the resource */
	assert(r->owner == current);

	/* Un-own this resource */
	r->owner = NULL;

	/* Let's wake up ONE waiter (if exists) that came first */
	if (!list_empty(&r->waitqueue)) {
		struct process *waiter = list_first_entry(&r->waitqueue, struct process, list);

		/**
		 * Ensure the waiter  is in the wait status
		 */
		assert(waiter->status == PROCESS_WAIT);

		/**
		 * Take out the waiter from the waiting queue. Note we use
		 * list_del_init() over list_del() to maintain the list head tidy
		 * (otherwise, the framework will complain on the list head
		 * when the process exits).
		 */
		list_del_init(&waiter->list);

		/* Update the process status */
		waiter->status = PROCESS_READY;

		/**
		 * Put the waiter process into ready queue. The framework will
		 * do the rest.
		 */
		list_add_tail(&waiter->list, &readyqueue);
	}
}



#include "sched.h"

/***********************************************************************
 * FIFO scheduler
 ***********************************************************************/
static int fifo_initialize(void)
{
	return 0;
}

static void fifo_finalize(void)
{
}

static struct process *fifo_schedule(void)
{
	struct process *next = NULL;

	if (!current || current->status == PROCESS_WAIT) {
		goto pick_next;
	}

	if (current->age < current->lifespan) {
		return current;
	}
	
pick_next:

	if (!list_empty(&readyqueue)) {

		next = list_first_entry(&readyqueue, struct process, list);

		list_del_init(&next->list);
	}

	return next;
}

struct scheduler fifo_scheduler = {
	.name = "FIFO",
	.acquire = fcfs_acquire,
	.release = fcfs_release,
	.initialize = fifo_initialize,
	.finalize = fifo_finalize,
	.schedule = fifo_schedule,
};


/***********************************************************************
 * SJF scheduler
 ***********************************************************************/
static struct process *sjf_schedule(void)
{
	struct process *next = NULL, *last = NULL;
	struct process *tmp1, *tmp2;

	if (!current || current->status == PROCESS_WAIT) goto pick_next;
	else if (current->age < current->lifespan) return current;
	else if (!list_empty(&readyqueue)) goto pick_next;
	return next;

pick_next:
	last = list_last_entry(&readyqueue, struct process, list);
	next = list_first_entry(&readyqueue, struct process, list);

	if (!list_empty(&readyqueue)) {
		list_for_each_entry_safe(tmp1, tmp2, &readyqueue, list) {
			if (tmp1->lifespan < next->lifespan) {
				next = tmp1;
			}

			if (tmp1 == last) {
				list_del_init(&next->list);
				return next;
			}
		}
	}
}

struct scheduler sjf_scheduler = {
	.name = "Shortest-Job First",
	.acquire = fcfs_acquire, /* Use the default FCFS acquire() */
	.release = fcfs_release, /* Use the default FCFS release() */
	.initialize = fifo_initialize,
	.finalize = fifo_finalize,
	.schedule = sjf_schedule,		 /* TODO: Assign sjf_schedule()
								to this function pointer to activate
								SJF in the system */
};


/***********************************************************************
 * SRTF scheduler
 ***********************************************************************/
static struct process *srft_schedule(void) {

	struct process *now = NULL, *last = NULL, *curr = NULL;
	struct process *tmp1, *tmp2;


	if (!current || current->status == PROCESS_WAIT) goto pick_next;
	else if (!list_empty(&readyqueue)) {
		if (current->age < current->lifespan) {
			curr = current;
			INIT_LIST_HEAD(&curr->list);
			list_add_tail(&curr->list, &readyqueue);
		}
		now = list_first_entry(&readyqueue, struct process, list);
		last = list_last_entry(&readyqueue, struct process, list);

		list_for_each_entry_safe(tmp1, tmp2, &readyqueue, list) {
			if (tmp1->lifespan - tmp1->age < now->lifespan - now->age) {
				now = tmp1;
			}
			
			if (tmp1 == last) {
				list_del_init(&now->list);
				return now;
			}
		}
	}
	else if(list_empty(&readyqueue) && (current->age != current->lifespan)) return current;
	else return NULL;

pick_next:
	if (!list_empty(&readyqueue)) {
		now = list_first_entry(&readyqueue, struct process, list);
		list_del_init(&now->list);
	}
	return now;
}

struct scheduler srtf_scheduler = {
	.name = "Shortest Remaining Time First",
	.acquire = fcfs_acquire, /* Use the default FCFS acquire() */
	.release = fcfs_release, /* Use the default FCFS release() */
	.initialize = fifo_initialize,
	.finalize = fifo_finalize,
	.schedule = srft_schedule,/* Obviously, you should implement srtf_schedule() and attach it here */
							 /* You need to check the newly created processes to implement SRTF.
							  * Use @forked() callback to mark newly created processes */
};


/***********************************************************************
 * Round-robin scheduler
 ***********************************************************************/
static struct process *rr_schedule(void) {
	struct process *now = NULL, *last = NULL, *curr = NULL;

	if (!current || current->status == PROCESS_WAIT) goto pick_next;
	else if (!list_empty(&readyqueue)) {
		if (current->age < current->lifespan) {
			curr = current;
			INIT_LIST_HEAD(&curr->list);
			list_add_tail(&curr->list, &readyqueue);
		}

		now = list_first_entry(&readyqueue, struct process, list);
		list_del_init(&now->list);
		return now;
	}
	else { // == if(list_empty(&readyqueue))
		if (current->age < current->lifespan) return current;
		return NULL;
	}

pick_next:
	if (!list_empty(&readyqueue)) {
		now = list_first_entry(&readyqueue, struct process, list);
		list_del_init(&now->list);
		return now;
	}
	return NULL;
}

struct scheduler rr_scheduler = {
	.name = "Round-Robin",
	.acquire = fcfs_acquire, /* Use the default FCFS acquire() */
	.release = fcfs_release, /* Use the default FCFS release() */
	.initialize = fifo_initialize,
	.finalize = fifo_finalize,
	.schedule = rr_schedule,	/* Obviously, you should implement rr_schedule() and attach it here */
};


/***********************************************************************
 * Priority scheduler
 ***********************************************************************/
bool prio_acquire(int resource_id) {
	struct resource *r = resources + resource_id;

	if (!r->owner) {
		r->owner = current;
		return true;
	}

	current->status = PROCESS_WAIT;

	list_add_tail(&current->list, &r->waitqueue);

	return false;
}

void prio_release(int resource_id) {
	struct resource *r = resources + resource_id;

	assert(r->owner == current);

	r->owner = NULL;

	if (!list_empty(&r->waitqueue)) {
		struct process *waiter = list_first_entry(&r->waitqueue, struct process, list);
		
		assert(waiter->status == PROCESS_WAIT);

		list_del_init(&waiter->list);

		waiter->status = PROCESS_READY;
		list_add_tail(&waiter->list, &readyqueue);
		
		
	}
}

static struct process *prio_schedule(void) {
	struct process *now = NULL, *last = NULL, *curr = NULL;
	struct process *tmp1, *tmp2;

	if (!list_empty(&readyqueue)) {
		if (current && (current->age < current->lifespan) && (current->status != PROCESS_WAIT)) {
			curr = current;
			INIT_LIST_HEAD(&curr->list);
			list_add_tail(&curr->list, &readyqueue);
		}
		now = list_first_entry(&readyqueue, struct process, list);
		last = list_last_entry(&readyqueue, struct process, list);

		list_for_each_entry_safe(tmp1, tmp2, &readyqueue, list) {
			if (tmp1->prio > now->prio) {
				now = tmp1;
			}

			if (tmp1 == last) {
				list_del_init(&now->list);
				return now;
			}
		}
	}
	else { // == if(list_empty(&readyqueue))
		if (current->age < current->lifespan) return current;
		return NULL;
	}
}

struct scheduler prio_scheduler = {
	.name = "Priority",
	.acquire = prio_acquire,
	.release = prio_release,
	.initialize = fifo_initialize,
	.finalize = fifo_finalize,
	.schedule = prio_schedule,
	/**
	 * Implement your own acqure/release function to make priority
	 * scheduler correct.
	 */
	/* Implement your own prio_schedule() and attach it here */
};


/***********************************************************************
 * Priority scheduler with priority inheritance protocol
 ***********************************************************************/
bool pip_acquire(int resource_id) {
	struct resource *r = resources + resource_id;
	struct process *now = NULL, *last = NULL;
	struct process *tmp1, *tmp2;

	if (!r->owner) {	// if owner is not existence
		r->owner = current;
		return true;
	}
	
	if (r->owner->pid != current->pid) {
		current->prio = r->owner->prio_orig;
		r->owner->prio = current->prio_orig;

		current->status = PROCESS_WAIT;
		list_add_tail(&current->list, &r->waitqueue);
		return false;
	}
	
	return true;
}

void pip_release(int resource_id) {
	struct resource *r = resources + resource_id;
	struct process *now = NULL, *last = NULL;
	struct process *tmp1, *tmp2;

	assert(r->owner == current);

	r->owner->prio = r->owner->prio_orig;
	r->owner = NULL;

	if (!list_empty(&r->waitqueue)) {
		list_for_each_entry_safe(tmp1, tmp2, &r->waitqueue, list) {
			list_del_init(&tmp1->list);
			list_add_tail(&tmp1->list, &readyqueue);
		}

	}
	if (!list_empty(&readyqueue)) {
		list_for_each_entry_safe(tmp1, tmp2, &readyqueue, list) {
			tmp1->prio = tmp1->prio_orig;
			tmp1->status = PROCESS_READY;
		}
	}
}

static struct process *pip_schedule(void) {
	struct process *now = NULL, *last = NULL, *curr = NULL;
	struct process *tmp1, *tmp2;

	if (!list_empty(&readyqueue)) {
		if (current && (current->age < current->lifespan) && (current->status != PROCESS_WAIT)) {
			curr = current;
			INIT_LIST_HEAD(&curr->list);
			list_add_tail(&curr->list, &readyqueue);
		}
		now = list_first_entry(&readyqueue, struct process, list);
		last = list_last_entry(&readyqueue, struct process, list);

		list_for_each_entry_safe(tmp1, tmp2, &readyqueue, list) {
			if (tmp1->prio > now->prio)	now = tmp1;

			if (tmp1 == last) {
				list_del_init(&now->list);
				return now;
			}
		}
	}
	else { // == if(list_empty(&readyqueue))
		if (current->age < current->lifespan) return current;
		return NULL;
	}

}

struct scheduler pip_scheduler = {
	.name = "Priority + Priority Inheritance Protocol",
	.acquire = pip_acquire,
	.release = pip_release,
	.initialize = fifo_initialize,
	.finalize = fifo_finalize,
	.schedule = pip_schedule,
	/**
	 * Implement your own acqure/release function too to make priority
	 * scheduler correct.
	 */
	/* It goes without saying to implement your own pip_schedule() */
};
